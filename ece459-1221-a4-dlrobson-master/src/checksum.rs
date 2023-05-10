use sha2::{Digest, Sha256};
use std::fmt;
#[derive(Default, Clone)]
pub struct Checksum {
    pub decoded: Vec<u8>,
}

impl Checksum {
    // Initialize the checksum with the SHA256 hash of the input string
    pub fn with_sha256(sha: &str) -> Self {
        let decoded = Sha256::digest(sha.as_bytes()).to_vec();
        Self { decoded }
    }

    // XOR the two checksums
    pub fn update(&mut self, rhs: &Self) {
        if self.decoded.is_empty() {
            self.decoded = rhs.decoded.clone();
        } else if rhs.decoded.is_empty() {
        } else {
            let a = &mut self.decoded;
            let b = &rhs.decoded;
            assert_eq!(a.len(), b.len());

            for (x, y) in a.iter_mut().zip(b) {
                *x ^= *y;
            }
        };
    }
}

impl fmt::Display for Checksum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.decoded.clone()))
    }
}
