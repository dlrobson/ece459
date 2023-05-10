use crate::packages::Dependency;
use crate::Packages;
use rpkg::debversion;

impl Packages {
    /// Gets the dependencies of package_name, and prints out whether they are satisfied (and by which library/version) or not.
    pub fn deps_available(&self, package_name: &str) {
        if !self.package_exists(package_name) {
            println!("no such package {}", package_name);
            return;
        }
        println!("Package {}:", package_name);

        let package_num = self.get_package_num(package_name);
        let dd = self.dependencies.get(package_num).unwrap();
        for dep in dd {
            // This prints alternative dependencies if there are any
            println!("- dependency {:?}", self.dep2str(dep));
            let satisfying_dep = match self.dep_is_satisfied(dep) {
                Some(d) => d,
                None => {
                    println!("-> not satisfied");
                    continue;
                }
            };
            // We have a satisifying dependency
            println!(
                "+ {} satisfied by installed version {}",
                satisfying_dep,
                self.get_installed_debver(satisfying_dep).unwrap()
            );
        }
    }

    /// Returns Some(package) which satisfies dependency dd, or None if not satisfied.
    pub fn dep_is_satisfied(&self, dd: &Dependency) -> Option<&str> {
        // If dependency is larger than one, that indicates that there are alternative packages.
        // We need to iterate over each
        for alt_d in dd {
            let package_name = self.get_package_name(alt_d.package_num);
            let installed_ver = match self.get_installed_debver(package_name) {
                Some(deb_version) => deb_version,
                _ => continue,
            };
            // The package was found. Let's see if it has a required version. If it doesn't, then
            // this is a satisfying dependency.
            let (required_rel, required_ver_str) = match &alt_d.rel_version {
                Some((rel, ver)) => (rel, ver),
                _ => return Some(package_name),
            };

            let valid_version = debversion::cmp_debversion_with_op(
                required_rel,
                installed_ver,
                &required_ver_str
                    .parse::<debversion::DebianVersionNum>()
                    .unwrap(),
            );

            if valid_version {
                return Some(package_name);
            }
        }
        return None;
    }

    /// Returns a Vec of packages which would satisfy dependency dd but for the version.
    /// Used by the how-to-install command, which calls compute_how_to_install().
    pub fn dep_satisfied_by_wrong_version(&self, dd: &Dependency) -> Vec<&str> {
        assert!(self.dep_is_satisfied(dd).is_none());

        let mut result = vec![];
        for alt_d in dd {
            let package_name = self.get_package_name(alt_d.package_num);

            // If it exists, then the first line "dep_is_satisfied" is none because the version
            // was incorrect
            if self.package_exists(package_name) {
                result.push(package_name)
            }
        }

        return result;
    }
}
