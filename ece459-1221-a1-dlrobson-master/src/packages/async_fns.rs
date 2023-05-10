use urlencoding::encode;

use curl::easy::{Easy2, Handler, WriteError};
use curl::multi::{Easy2Handle, Multi};
use std::str;
// use std::sync::atomic::{AtomicI32, Ordering};
use std::time::Duration;

use crate::Packages;

struct Collector(Box<String>);
impl Handler for Collector {
    fn write(&mut self, data: &[u8]) -> Result<usize, WriteError> {
        (*self.0).push_str(str::from_utf8(&data.to_vec()).unwrap());
        Ok(data.len())
    }
}

const DEFAULT_SERVER: &str = "ece459.patricklam.ca:4590";
impl Drop for Packages {
    fn drop(&mut self) {
        self.execute()
    }
}

// static EASYKEY_COUNTER: AtomicI32 = AtomicI32::new(0);

pub struct AsyncState {
    server: String,
    multi: Multi,
    easy_handles: Vec<(Easy2Handle<Collector>, String, String, String)>,
}

impl AsyncState {
    pub fn new() -> AsyncState {
        AsyncState {
            server: String::from(DEFAULT_SERVER),
            multi: Multi::new(),
            easy_handles: Vec::new(),
        }
    }
}

impl Packages {
    pub fn set_server(&mut self, new_server: &str) {
        self.async_state.server = String::from(new_server);
    }

    /// Retrieves the version number of pkg and calls enq_verify_with_version with that version number.
    pub fn enq_verify(&mut self, pkg: &str) {
        let version = self.get_available_debver(pkg);
        match version {
            None => {
                println!("Error: package {} not defined.", pkg);
                return;
            }
            Some(v) => {
                let vs = &v.to_string();
                self.enq_verify_with_version(pkg, vs);
            }
        };
    }

    /// Enqueues a request for the provided version/package information. Stores any needed state to
    /// async_state so that execute() can handle the results and print out needed output.
    pub fn enq_verify_with_version(&mut self, pkg: &str, version: &str) {
        let url = format!(
            "http://{}/rest/v1/checksums/{}/{}",
            self.async_state.server,
            encode(pkg),
            encode(version)
        );
        println!("queueing request {}", url);
        let mut easy = Easy2::new(Collector(Box::new(String::new())));
        easy.get(true).unwrap();
        easy.url(&url).unwrap();
        let handle = self.async_state.multi.add2(easy);

        self.async_state.easy_handles.push((
            handle.unwrap(),
            (*pkg).to_string(),
            (*version).to_string(),
            self.get_md5sum(pkg).unwrap().trim().to_string(),
        ));
    }

    /// Asks curl to perform all enqueued requests. For requests that succeed with response code
    /// 200, compares received MD5sum with local MD5sum (perhaps stored earlier). For requests
    /// that fail with 400+, prints error message.
    pub fn execute(&mut self) {
        let async_state_ref = &mut self.async_state;
        while async_state_ref.multi.perform().unwrap() > 0 {
            async_state_ref
                .multi
                .wait(&mut [], Duration::from_secs(1))
                .unwrap();
        }
        for (handle, pkg, version, md5sum_database) in async_state_ref.easy_handles.drain(..) {
            let mut easy_result = async_state_ref.multi.remove2(handle).unwrap();
            match easy_result.response_code().unwrap() {
                200 => {
                    let md5sum = (*easy_result.get_ref().0).trim().to_string();
                    let md5sum_same = md5sum == md5sum_database;
                    println!("verifying {}, matches: {:?}", pkg, md5sum_same);
                }
                c => {
                    println!(
                        "got error {} on request for package {} version {}",
                        c, pkg, version
                    );
                }
            };
        }
    }
}
