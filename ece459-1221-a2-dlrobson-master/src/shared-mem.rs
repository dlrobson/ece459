// Starter code for ECE 459 Lab 2, Winter 2021

// YOU SHOULD MODIFY THIS FILE TO USE THREADING AND SHARED MEMORY

#![warn(clippy::all)]
use hmac::{Hmac, Mac, NewMac};
use sha2::Sha256;
use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

const DEFAULT_ALPHABETS: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";

type HmacSha256 = Hmac<Sha256>;

// Check if a JWT secret is correct
fn is_secret_valid(msg: &[u8], sig: &[u8], secret: &[u8]) -> bool {
    let mut mac = HmacSha256::new_varkey(secret).unwrap();
    mac.update(msg);
    mac.verify(sig).is_ok()
}

// Contextual info for solving a JWT
#[derive(Clone)]
struct JwtSolver {
    alphabet: Vec<u8>, // set of possible bytes in the secret
    max_len: usize,    // max length of the secret
    msg: Vec<u8>,      // JWT message
    sig64: Vec<u8>,    // JWT signature (base64 decoded)
}

impl JwtSolver {
    // Recursively check every possible secret string,
    // returning the correct secret if it exists
    fn check_all(&self, secret: Vec<u8>, is_finished: &mut Arc<AtomicBool>) -> Option<Vec<u8>> {
        if is_secret_valid(&self.msg, &self.sig64, &secret) {
            return Some(secret); // found it!
        }

        if secret.len() == self.max_len || is_finished.load(Ordering::Relaxed) {
            return None; // no secret of length <= max_len
        }

        for &c in self.alphabet.iter() {
            // allocate space for a secret one character longer
            let mut new_secret = Vec::with_capacity(secret.len() + 1);
            // build the new secret
            new_secret.extend(secret.iter().chain(&mut [c].iter()));
            // check this secret, and recursively check longer ones
            if let Some(ans) = self.check_all(new_secret, &mut is_finished.clone()) {
                is_finished.store(true, Ordering::Relaxed);
                return Some(ans);
            }
        }
        None
    }
    fn shared_memory_solver(&self) -> Option<Vec<u8>> {
        // Number of characters in each thread
        let cpu_count = num_cpus::get() - 1;
        let secret_found = Arc::new(AtomicBool::new(false));

        // The following steps evenly divides the letters between the cpu's
        let min_chars = self.alphabet.len() / cpu_count;
        let num_threads_with_extra = self.alphabet.len() % cpu_count;
        let mut thread_tasks: Vec<Vec<u8>> = vec![];
        for i in 0..num_threads_with_extra {
            let contents = self.alphabet[i * (min_chars + 1)..(i + 1) * (min_chars + 1)].to_vec();
            thread_tasks.push(contents);
        }
        let offset = num_threads_with_extra * (min_chars + 1);

        for i in 0..(cpu_count - num_threads_with_extra) {
            let contents =
                self.alphabet[(offset + i * min_chars)..(offset + (i + 1) * min_chars)].to_vec();
            thread_tasks.push(contents);
        }

        let mut handles = Vec::with_capacity(cpu_count);
        for thread_i in 0..cpu_count {
            let solver_copy: JwtSolver = self.clone();
            let thread_task = thread_tasks[thread_i].clone();
            let mut secret_found_clone = secret_found.clone();
            handles.push(thread::spawn(move || {
                execute_thread(solver_copy, thread_task, &mut secret_found_clone)
            }));
        }
        let mut return_val: Option<Vec<u8>> = None;
        for handle in handles {
            match handle.join().unwrap() {
                Some(msg) => {
                    return_val = Some(msg);
                }
                _ => {
                    continue;
                }
            };
        }
        return_val
    }
}

fn execute_thread(
    solver: JwtSolver,
    thread_task: Vec<u8>,
    secret_found: &mut Arc<AtomicBool>,
) -> Option<Vec<u8>> {
    for s in thread_task {
        let vect: Vec<u8> = vec![s];
        match solver.check_all(vect, &mut secret_found.clone()) {
            Some(msg) => {
                return Some(msg);
            }
            _ => {
                if secret_found.load(Ordering::Relaxed) {
                    break;
                }
                continue;
            }
        };
    }
    None
}

fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.len() < 3 {
        eprintln!("Usage: <token> <max_len> [alphabet]");
        return;
    }
    let token = &args[1];

    let max_len = match args[2].parse::<u32>() {
        Ok(len) => len,
        Err(_) => {
            eprintln!("Invalid max length");
            return;
        }
    };

    let alphabet = args
        .get(3)
        .map(|a| a.as_bytes())
        .unwrap_or(DEFAULT_ALPHABETS)
        .into();

    // find index of last '.'
    let dot = match token.rfind('.') {
        Some(pos) => pos,
        None => {
            eprintln!("No dot found in token");
            return;
        }
    };

    // message is everything before the last dot
    let msg = token.as_bytes()[..dot].to_vec();
    // signature is everything after the last dot
    let sig = &token.as_bytes()[dot + 1..];

    // convert base64 encoding into binary
    let sig64 = match base64::decode_config(sig, base64::URL_SAFE_NO_PAD) {
        Ok(sig) => sig,
        Err(_) => {
            eprintln!("Invalid signature");
            return;
        }
    };

    // build the solver and run it to get the answer
    let solver = JwtSolver {
        alphabet,
        max_len: max_len as usize,
        msg,
        sig64,
    };
    let ans = solver.shared_memory_solver();

    match ans {
        Some(ans) => println!(
            "{}",
            std::str::from_utf8(&ans).expect("answer not a valid string")
        ),
        None => println!("No answer found"),
    };
}
