//! Tooling to check the correctness of `Drop` implementations.
//!
//! Properly testing a container type like `Vec<T>` requires verifying that every value in the
//! container is neither leaked, nor dropped multiple times.
//!
//! To detect leaks, this crate provides a `DropToken` type whose drop implementation sets a flag in a
//! `DropState` with interior mutability (specifically atomics). Secondly, these states are stored
//! in a `DropCheck` set. If any any token hasn't been dropped when the `DropCheck` is dropped, the
//! `DropCheck`'s drop impl panics:
//!
//! ```should_panic
//! # use dropcheck::DropCheck;
//! let dropcheck = DropCheck::new();
//! let token = dropcheck.token();
//!
//! std::mem::forget(token); // leaked!
//! // panics when dropcheck goes out of scope
//! ```
//!
//! Secondly, dropping a token twice panics:
//!
//! ```should_panic
//! # use dropcheck::DropCheck;
//! let dropcheck = DropCheck::new();
//! let mut token = dropcheck.token();
//!
//! unsafe {
//!     std::ptr::drop_in_place(&mut token);
//!     std::ptr::drop_in_place(&mut token); // panics
//! }
//! ```

use std::fmt;
use std::sync::{Arc, Weak, RwLock, atomic::{AtomicUsize, Ordering}};

/// A drop-checking token.
///
/// Created by `DropCheck`.
#[derive(Debug)]
pub struct DropToken {
    set: Weak<RwLock<Vec<Arc<DropState>>>>,
    state: Arc<DropState>,
}

impl Drop for DropToken {
    fn drop(&mut self) {
        self.state.set_dropped();
    }
}

/// Cloning a `DropToken` creates a fresh state, that's still tied to the `DropCheck` set that
/// created the token. This means that leaking the cloned token is detected:
///
/// ```should_panic
/// # use dropcheck::DropCheck;
/// let dropcheck = DropCheck::new();
/// let token = dropcheck.token();
///
/// let cloned_token = token.clone();
/// std::mem::forget(cloned_token);
/// // panics when dropcheck is dropped
/// ```
///
/// Since the new token is part of the set it came from, it affects `none_dropped`/`all_dropped`:
///
/// ```
/// # use dropcheck::DropCheck;
/// let dropcheck = DropCheck::new();
/// let token = dropcheck.token();
///
/// let cloned_token = token.clone();
/// assert!(dropcheck.none_dropped());
///
/// drop(cloned_token);
/// assert!(!dropcheck.none_dropped());
/// ```
impl Clone for DropToken {
    fn clone(&self) -> Self {
        let state = DropState::new();
        if let Some(set) = self.set.upgrade() {
            set.write().unwrap().push(Arc::clone(&state));
            Self {
                set: Arc::downgrade(&set),
                state,
            }
        } else {
            Self {
                set: Weak::new(),
                state,
            }
        }
    }
}

/// The state of a particular `DropToken`.
pub struct DropState {
    count: AtomicUsize,
}

impl fmt::Debug for DropState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct(&format!("DropState<{:p}>", self))
            .field("count", &self.count)
            .finish()
    }
}

impl Drop for DropState {
    fn drop(&mut self) {
        match self.count.get_mut() {
            1 => {},
            0 => panic!("token not dropped"),
            _ => panic!("invalid drop count: {}"),
        }
    }
}

impl DropState {
    /// Returns true if the token associated with this state has been dropped.
    pub fn is_dropped(&self) -> bool {
        !self.is_not_dropped()
    }

    /// The inverse of `is_dropped()`.
    pub fn is_not_dropped(&self) -> bool {
        match self.count.load(Ordering::SeqCst) {
            0 => true,
            1 => false,
            x => panic!("invalid drop count: {}", x),
        }
    }

    fn new() -> Arc<Self> {
        Arc::new(Self { count: AtomicUsize::new(0) })
    }

    fn set_dropped(&self) {
        match self.count.swap(1, Ordering::SeqCst) {
            0 => {},
            1 => panic!("already dropped"),
            x => panic!("invalid drop count: {}", x),
        }
    }
}

/// A set of `DropToken`'s.
#[derive(Debug, Default)]
pub struct DropCheck {
    set: Arc<RwLock<Vec<Arc<DropState>>>>,
}

impl Drop for DropCheck {
    fn drop(&mut self) {
        assert!(self.all_dropped(), "not all tokens dropped");
    }
}

impl DropCheck {
    /// Creates a new `DropCheck` set.
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&self, state: Arc<DropState>) {
        self.set.write().unwrap().push(state)
    }

    /// Creates a new `DropToken`, whose state is part of this set.
    pub fn token(&self) -> DropToken {
        let state = DropState::new();
        self.push(Arc::clone(&state));

        DropToken {
            set: Arc::downgrade(&self.set),
            state,
        }
    }

    /// Creates a new `DropToken`, and also gives you a handle to the state.
    ///
    /// # Examples
    ///
    /// Checking when an operation drops a value:
    ///
    /// ```
    /// # use dropcheck::DropCheck;
    /// let dropcheck = DropCheck::new();
    ///
    /// let mut v = vec![dropcheck.token(); 10];
    ///
    /// let (t1, s1) = dropcheck.pair();
    /// v.push(t1);
    ///
    /// assert!(s1.is_not_dropped());
    /// v.pop();
    /// assert!(s1.is_dropped()); // vec drops items immediately
    /// ```
    pub fn pair(&self) -> (DropToken, Arc<DropState>) {
        let state = DropState::new();
        self.push(Arc::clone(&state));

        (DropToken {
            set: Arc::downgrade(&self.set),
            state: Arc::clone(&state),
        }, state)
    }

    /// Returns true if none of the `Token`s in this set have been dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dropcheck::DropCheck;
    /// let set = DropCheck::new();
    /// assert!(set.none_dropped()); // an empty set has no dropped tokens
    ///
    /// let t1 = set.token();
    /// assert!(set.none_dropped());
    ///
    /// let t2 = set.token();
    /// assert!(set.none_dropped());
    ///
    /// drop(t1);
    /// assert!(!set.none_dropped());
    /// ```
    pub fn none_dropped(&self) -> bool {
        self.set.read().unwrap()
            .iter().all(|state| state.is_not_dropped())
    }

    /// Returns true if all of the `Token`s have been dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dropcheck::DropCheck;
    /// let set = DropCheck::new();
    /// assert!(set.all_dropped()); // all of the tokens in an empty set have been dropped
    ///
    /// let mut v = vec![];
    /// for _ in 0 .. 100 {
    ///     v.push(set.token());
    /// }
    /// assert!(!set.all_dropped());
    ///
    /// drop(v);
    /// assert!(set.all_dropped()); // vec has dropped every token in it
    /// ```
    pub fn all_dropped(&self) -> bool {
        self.set.read().unwrap()
            .iter().all(|state| state.is_dropped())
    }
}
