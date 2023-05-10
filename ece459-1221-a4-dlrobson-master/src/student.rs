use super::{checksum::Checksum, idea::Idea, package::Package, Event};
use crossbeam::channel::Receiver;
use std::collections::HashMap;
use std::io::{stdout, Write};
use std::sync::{Arc, Mutex};

pub struct Student {
    id: usize,
    acquired_idea: Option<Idea>,
    acquired_packages: Vec<Package>,
    idea_recv: Receiver<Event>,
    package_recv: Receiver<Event>,
    packages_used: Vec<usize>,
    ideas_used: Vec<Checksum>,
}

impl Student {
    pub fn new(id: usize, idea_recv: Receiver<Event>, package_recv: Receiver<Event>) -> Self {
        Self {
            id,
            idea_recv,
            package_recv,
            acquired_idea: None,
            acquired_packages: vec![],
            packages_used: vec![],
            ideas_used: vec![],
        }
    }

    fn build_idea(&mut self, idea_checksum_map: &Arc<Mutex<HashMap<String, Checksum>>>) {
        if let Some(ref idea) = self.acquired_idea {
            // Can only build ideas if we have acquired sufficient packages
            let pkgs_required = idea.num_pkg_required;

            if pkgs_required > self.acquired_packages.len() {
                return;
            }

            // Update idea and package checksums
            // All of the packages used in the update are deleted, along with the idea
            let idea_checksum;
            {
                idea_checksum = idea_checksum_map
                    .lock()
                    .unwrap()
                    .get(&idea.name)
                    .unwrap()
                    .clone();
            }
            self.ideas_used.push(idea_checksum);

            let mut output: String = format!(
                "\nStudent {} built {} using {} packages\n",
                self.id, idea.name, pkgs_required
            );

            for pkg in self.acquired_packages.drain(0..pkgs_required) {
                output.push_str(format!("> {}\n", &pkg.name).as_str());
                self.packages_used.push(pkg.index);
            }

            // We want the subsequent prints to be together, so we lock stdout
            let stdout = stdout();
            {
                let mut handle = stdout.lock();
                match write!(handle, "{}", output) {
                    Err(e) => println!("{:?}", e),
                    _ => (),
                };
            }

            self.acquired_idea = None;
        }
    }

    fn empty_jobs(
        &mut self,
        global_package_checksum_prot: &Arc<Mutex<Checksum>>,
        global_idea_checksum_prot: &Arc<Mutex<Checksum>>,
        package_checksums_prot: &Arc<Mutex<Vec<Checksum>>>,
    ) {
        {
            let mut global_package_checksum = global_package_checksum_prot.lock().unwrap();
            let package_checksums = package_checksums_prot.lock().unwrap();
            for pkg_i in self.packages_used.drain(0..self.packages_used.len()) {
                global_package_checksum.update(&package_checksums[pkg_i]);
            }
        }
        {
            let mut global_idea_checksum = global_idea_checksum_prot.lock().unwrap();
            for c in self.ideas_used.drain(0..self.ideas_used.len()) {
                global_idea_checksum.update(&c);
            }
        }
    }

    pub fn run(
        &mut self,
        global_idea_checksum_prot: Arc<Mutex<Checksum>>,
        global_package_checksum_prot: Arc<Mutex<Checksum>>,
        idea_checksum_map: Arc<Mutex<HashMap<String, Checksum>>>,
        package_checksum_vec_prot: Arc<Mutex<Vec<Checksum>>>,
    ) {
        loop {
            // If the student is not working on an idea, then they will take the new idea
            // and attempt to build it. Otherwise, the idea is skipped.
            if self.acquired_idea.is_none() {
                let idea_event;
                match self.idea_recv.try_recv() {
                    Ok(e) => idea_event = e,
                    Err(_) => {
                        self.empty_jobs(
                            &global_package_checksum_prot,
                            &global_idea_checksum_prot,
                            &package_checksum_vec_prot,
                        );
                        continue;
                    }
                };

                match idea_event {
                    Event::NewIdea(idea) => {
                        self.acquired_idea = Some(idea);
                        self.build_idea(&idea_checksum_map);
                    }

                    Event::OutOfIdeas => {
                        self.empty_jobs(
                            &global_package_checksum_prot,
                            &global_idea_checksum_prot,
                            &package_checksum_vec_prot,
                        );
                        return;
                    }

                    Event::DownloadComplete(_) => (),
                }
            } else {
                // This is the case where we ARE working on an idea. We wait to receive a
                // package and then attempt to build the idea
                let pkg_event;
                match self.package_recv.try_recv() {
                    Ok(e) => pkg_event = e,
                    Err(_) => {
                        self.empty_jobs(
                            &global_package_checksum_prot,
                            &global_idea_checksum_prot,
                            &package_checksum_vec_prot,
                        );
                        continue;
                    }
                };
                match pkg_event {
                    Event::DownloadComplete(pkg) => {
                        // Getting a new package means the current idea may now be buildable, so the
                        // student attempts to build it
                        self.acquired_packages.push(pkg);
                        self.build_idea(&idea_checksum_map);
                    }
                    Event::NewIdea(_) => (),
                    Event::OutOfIdeas => (),
                }
            }
        }
    }
}
