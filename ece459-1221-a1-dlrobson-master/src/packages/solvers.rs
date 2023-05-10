use crate::packages::Dependency;
use crate::packages::RelVersionedPackageNum;
use crate::Packages;
use rpkg::debversion;

impl Packages {
    /// Computes a solution for the transitive dependencies of package_name; when there is a choice
    /// A | B | C, chooses the first option A. Returns a Vec<i32> of package numbers.
    ///
    /// Note: does not consider which packages are installed.
    pub fn transitive_dep_solution(&self, package_name: &str) -> Vec<i32> {
        if !self.package_exists(package_name) {
            return vec![];
        }
        // List of dependencies that need to be satisfied
        let deps: &Vec<Dependency> = &*self
            .dependencies
            .get(self.get_package_num(package_name))
            .unwrap();
        let mut dependency_set = vec![];

        // Load the initial list of dependencies into the dependency_set
        for dep in deps {
            // Always select the first package, hence the dereference to the first element
            dependency_set.push(dep[0].package_num)
        }

        let mut prev_dep_count = 0;
        let mut new_dep_count = dependency_set.len();

        while prev_dep_count != new_dep_count {
            for dep_i in prev_dep_count..new_dep_count {
                // Current dependency we are checking
                let dep_package_num = dependency_set[dep_i];
                // Find the dependencies for the current dependency, and verify it's not already
                // within the vector before adding it in
                let sub_deps: &Vec<Dependency> = &*self.dependencies.get(&dep_package_num).unwrap();
                for sub_dep in sub_deps {
                    // Always select the first package, hence the dereference to the first element
                    let sub_dep_package_num = sub_dep[0].package_num;
                    if !dependency_set.contains(&sub_dep_package_num) {
                        dependency_set.push(sub_dep_package_num)
                    }
                }
            }
            prev_dep_count = new_dep_count;
            new_dep_count = dependency_set.len();
        }
        return dependency_set;
    }

    /// Computes a set of packages that need to be installed to satisfy package_name's deps given
    /// the current installed packages. When a dependency A | B | C is unsatisfied, there are two
    /// possible cases:
    ///   (1) there are no versions of A, B, or C installed; pick the alternative with the highest
    ///       version number (yes, compare apples and oranges).
    ///   (2) at least one of A, B, or C is installed (say A, B), but with the wrong version; of
    ///       the installed packages (A, B), pick the one with the highest version number.
    pub fn compute_how_to_install(&self, package_name: &str) -> Vec<i32> {
        if !self.package_exists(package_name) {
            return vec![];
        }
        // List of dependencies that need to be satisfied
        let deps: &Vec<Dependency> = &*self
            .dependencies
            .get(self.get_package_num(package_name))
            .unwrap();
        let mut dependencies_to_add: Vec<i32> = vec![];
        // Add dependencies that are not met
        for dep in deps {
            // This dependency is satisfied, and does not need to be installed
            if !self.dep_is_satisfied(dep).is_none() {
                continue;
            }
            let s_package_num = self.satisfying_package(dep).unwrap();
            if dependencies_to_add.contains(&s_package_num) {
                continue;
            }
            dependencies_to_add.push(s_package_num);
        }

        let mut prev_dep_count = 0;
        let mut new_dep_count = dependencies_to_add.len();
        while prev_dep_count != new_dep_count {
            for dep_i in prev_dep_count..new_dep_count {
                // Current dependency we are checking
                let dep_num = dependencies_to_add[dep_i];
                // Find the dependencies for the current dependency
                let sub_deps: &Vec<Dependency> = &*self.dependencies.get(&dep_num).unwrap();
                for sub_dep in sub_deps {
                    if !self.dep_is_satisfied(sub_dep).is_none() {
                        continue;
                    }
                    let s_package_num = self.satisfying_package(sub_dep).unwrap();

                    if dependencies_to_add.contains(&s_package_num) {
                        continue;
                    }
                    dependencies_to_add.push(s_package_num);
                }
            }
            prev_dep_count = new_dep_count;
            new_dep_count = dependencies_to_add.len();
        }

        return dependencies_to_add;
    }
    /// Now run the loop. In order:
    ///  (1) If the sub-dependency is installed, do nothing
    ///  (2) If the package is installed but it's the wrong version:
    ///      (a) If only one dependency exists with the wrong version, install that one
    ///      (b) If more than one dependency exists with the wrong version, install the one with
    ///          the higher version number
    ///  (3) If the package is not installed:
    ///      (a) If there is a single dependency, install that version
    ///      (b) If there is more than one dependency, install the version with the higher
    ///          version number
    pub fn satisfying_package(&self, dd: &Dependency) -> Result<i32, &str> {
        assert!(self.dep_is_satisfied(dd).is_none());
        if dd.len() == 1 {
            return Ok(dd[0].package_num);
        }

        let incorrect_version_dep_names = self.dep_satisfied_by_wrong_version(dd);
        if incorrect_version_dep_names.len() == 1 {
            return Ok(*self.get_package_num(incorrect_version_dep_names[0]));
        }

        if incorrect_version_dep_names.len() > 1 {
            let mut largest_ver_dep: &RelVersionedPackageNum = dd
                .iter()
                .find(|dep| {
                    incorrect_version_dep_names[0] == self.get_package_name(dep.package_num)
                })
                .unwrap();

            for i in 1..incorrect_version_dep_names.len() {
                let package_name = incorrect_version_dep_names[i];
                let dep: &RelVersionedPackageNum = dd
                    .iter()
                    .find(|dep| package_name == self.get_package_name(dep.package_num))
                    .unwrap();

                // If there is no release version on this dependency, skip it.
                if dep.rel_version.is_none() {
                    continue;
                }
                // This means the current dep has no release version. Automatically update
                if largest_ver_dep.rel_version.is_none() {
                    largest_ver_dep = dep;
                    continue;
                }
                let first_ver: debversion::DebianVersionNum = largest_ver_dep
                    .rel_version
                    .as_ref()
                    .unwrap()
                    .1
                    .parse::<debversion::DebianVersionNum>()
                    .unwrap();
                let second_ver: debversion::DebianVersionNum = dep
                    .rel_version
                    .as_ref()
                    .unwrap()
                    .1
                    .parse::<debversion::DebianVersionNum>()
                    .unwrap();

                if debversion::cmp_debversion_with_op(
                    &debversion::VersionRelation::StrictlyGreater,
                    &first_ver,
                    &second_ver,
                ) {
                    largest_ver_dep = dep;
                }
            }
            return Ok(largest_ver_dep.package_num);
        }

        // There is no incorrect version dependencies install by this point
        let mut largest_ver_dep: &RelVersionedPackageNum = &dd[0];

        for i in 1..dd.len() {
            let dep: &RelVersionedPackageNum = &dd[i];

            // If there is no release version on this dependency, skip it.
            if dep.rel_version.is_none() {
                continue;
            }
            // This means the current dep has no release version. Automatically update
            if largest_ver_dep.rel_version.is_none() {
                largest_ver_dep = dep;
                continue;
            }
            let first_ver: debversion::DebianVersionNum = largest_ver_dep
                .rel_version
                .as_ref()
                .unwrap()
                .1
                .parse::<debversion::DebianVersionNum>()
                .unwrap();
            let second_ver: debversion::DebianVersionNum = dep
                .rel_version
                .as_ref()
                .unwrap()
                .1
                .parse::<debversion::DebianVersionNum>()
                .unwrap();

            if debversion::cmp_debversion_with_op(
                &debversion::VersionRelation::StrictlyGreater,
                &first_ver,
                &second_ver,
            ) {
                largest_ver_dep = dep;
            }
        }
        return Ok(largest_ver_dep.package_num);
    }
}
