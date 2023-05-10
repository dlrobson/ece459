use super::checksum::Checksum;
use super::Event;
use crossbeam::channel::Sender;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct Package {
    pub name: String,
    pub index: usize,
}

pub struct PackageDownloader {
    pkg_start_idx: usize,
    num_pkgs: usize,
    event_sender: Sender<Event>,
}

impl PackageDownloader {
    pub fn new(pkg_start_idx: usize, num_pkgs: usize, event_sender: Sender<Event>) -> Self {
        Self {
            pkg_start_idx,
            num_pkgs,
            event_sender,
        }
    }

    pub fn run(
        &self,
        global_package_checksum_prot: Arc<Mutex<Checksum>>,
        package_checksum_vec_prot: Arc<Mutex<Vec<Checksum>>>,
        package_names_vec: Arc<Mutex<String>>,
    ) {
        let package_names_str = package_names_vec.lock().unwrap().clone();
        let package_names: Vec<String> = package_names_str
            .split("\n")
            .map(|s| s.to_string())
            .collect();

        {
            let mut packages_vec = package_checksum_vec_prot.lock().unwrap();
            if packages_vec.is_empty() {
                packages_vec.resize(package_names.len(), Checksum::default())
            }
        }

        let mut global_checksum_updates: Vec<Checksum> = vec![];

        // Generate a set of packages and place them into the event queue
        // Update the package checksum with each package name
        for i in 0..self.num_pkgs {
            let index = (self.pkg_start_idx + i) % package_names.len();
            let name = package_names[index].clone();

            let checksum: Checksum;
            {
                let mut package_checksum_vec = package_checksum_vec_prot.lock().unwrap();
                if package_checksum_vec[index].decoded.len() < 1 {
                    let c = Checksum::with_sha256(&name);
                    {
                        package_checksum_vec[index] = c.clone();
                    }
                    checksum = c;
                } else {
                    checksum = package_checksum_vec[index].clone();
                }
            }

            global_checksum_updates.push(checksum.clone());

            self.event_sender
                .send(Event::DownloadComplete(Package {
                    name: name,
                    index: index,
                }))
                .unwrap();
        }

        {
            let mut global_checksum = global_package_checksum_prot.lock().unwrap();
            for c in global_checksum_updates.iter() {
                global_checksum.update(&c);
            }
        }
    }
}
