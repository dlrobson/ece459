use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

use regex::Regex;

use crate::packages::{Dependency, RelVersionedPackageNum};
use crate::Packages;

use rpkg::debversion;

const KEYVAL_REGEX: &str = r"(?P<key>(\w|-)+): (?P<value>.+)";
const PKGNAME_AND_VERSION_REGEX: &str =
    r"(?P<pkg>(\w|\.|\+|-)+)( \((?P<op>(<|=|>)(<|=|>)?) (?P<ver>.*)\))?";

impl Packages {
    /// Loads packages and version numbers from a file, calling get_package_num_inserting on the package name
    /// and inserting the appropriate value into the installed_debvers map with the parsed version number.
    pub fn parse_installed(&mut self, filename: &str) {
        let kv_regexp = Regex::new(KEYVAL_REGEX).unwrap();

        let lines = match read_lines(filename) {
            Err(_) => {
                println!("Input not parsable. No installed packages found.");
                return;
            }
            Ok(lines) => lines,
        };

        let mut current_package_num = 0;
        for line in lines {
            // If the line had did not have a valid match, continue to the next line
            let ip = match line {
                Err(_) => continue,
                Ok(line) => line,
            };

            // If no match, continue
            let line_data = match kv_regexp.captures(&ip) {
                Some(line_data) => line_data,
                _ => continue,
            };

            // We have a match with data. We only parse if the key is Package or Version.
            match line_data.name("key").unwrap().as_str() {
                "Package" => {
                    let package_name = line_data.name("value").unwrap().as_str().trim();
                    current_package_num = self.get_package_num_inserting(package_name);
                }
                "Version" => {
                    let deb_version = line_data
                        .name("value")
                        .unwrap()
                        .as_str()
                        .trim()
                        .parse::<debversion::DebianVersionNum>()
                        .unwrap();
                    self.installed_debvers
                        .insert(current_package_num, deb_version);
                }
                _ => continue,
            };
        }
        println!(
            "Packages installed: {}",
            self.installed_debvers.keys().len()
        );
    }

    /// Loads packages, version numbers, dependencies, and md5sums from a file, calling get_package_num_inserting on the package name
    /// and inserting the appropriate values into the dependencies, md5sum, and available_debvers maps.
    pub fn parse_packages(&mut self, filename: &str) {
        let kv_regexp = Regex::new(KEYVAL_REGEX).unwrap();
        let pkgver_regexp = Regex::new(PKGNAME_AND_VERSION_REGEX).unwrap();

        let lines = match read_lines(filename) {
            Err(_) => {
                println!("Input not parsable. No installed packages found.");
                return;
            }
            Ok(lines) => lines,
        };

        // Need this variable to store the key to the hashmap
        let mut current_package_num = 0;
        for line in lines {
            // If the line had did not have a valid match, continue to the next line
            let ip = match line {
                Err(_) => continue,
                Ok(line) => line,
            };

            // If no match, continue
            let caps = match kv_regexp.captures(&ip) {
                Some(caps) => caps,
                _ => continue,
            };

            match caps.name("key").unwrap().as_str() {
                "Package" => {
                    let package_name = caps.name("value").unwrap().as_str();
                    current_package_num = self.get_package_num_inserting(package_name);
                }
                "MD5sum" => {
                    let md5sum_value = caps.name("value").unwrap().as_str();
                    self.md5sums
                        .insert(current_package_num, md5sum_value.to_string());
                }
                "Version" => {
                    let debver = caps
                        .name("value")
                        .unwrap()
                        .as_str()
                        .trim()
                        .parse::<debversion::DebianVersionNum>()
                        .unwrap();
                    self.available_debvers.insert(current_package_num, debver);
                }
                "Depends" => {
                    // If there already exist depends for this package, it's been previously
                    // installed. Skip this one
                    if !self
                        .dependencies
                        .get(&current_package_num)
                        .unwrap()
                        .is_empty()
                    {
                        continue;
                    }
                    // Iterate through each dependency that's been ',' spliced
                    for depend in caps.name("value").unwrap().as_str().split(',') {
                        // Holds a dependency for the main package to be installed
                        let mut main_dependency: Dependency = Vec::new();
                        // These are alternative dependencies
                        for alt_depend in depend.split('|') {
                            // If a package match was found, append it to
                            // main_dependency. Each of these packages within this loop
                            // must be appended
                            let dep_caps = match pkgver_regexp.captures(alt_depend) {
                                Some(dep_caps) => dep_caps,
                                _ => continue,
                            };

                            // Create an instance of the next dependency to be
                            // appendeds to the dependency map
                            let pkg = dep_caps.name("pkg").unwrap().as_str();
                            let dep_package_num = self.get_package_num_inserting(pkg);
                            let mut package_info = RelVersionedPackageNum {
                                package_num: dep_package_num,
                                rel_version: None,
                            };
                            // This covers the instance where the package does not have
                            // a version number.
                            if let (Some(op), Some(ver)) =
                                (dep_caps.name("op"), dep_caps.name("ver"))
                            {
                                package_info.rel_version = Some((
                                    op.as_str().parse::<debversion::VersionRelation>().unwrap(),
                                    ver.as_str().to_string(),
                                ))
                            }
                            main_dependency.push(package_info);
                        }
                        if let Some(depend_ref) = self.dependencies.get_mut(&current_package_num) {
                            depend_ref.push(main_dependency);
                        }
                    }
                }
                _ => (),
            }
        }
        println!(
            "Packages available: {}",
            self.available_debvers.keys().len()
        );
    }
}

// standard template code downloaded from the Internet somewhere
fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}
