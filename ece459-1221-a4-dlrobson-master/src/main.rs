#![warn(clippy::all)]
use crossbeam::channel::{unbounded, Receiver, Sender};
use lab4::{
    checksum::Checksum, idea::IdeaGenerator, package::PackageDownloader, student::Student, Event,
};
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::sync::{Arc, Mutex};
use std::thread::spawn;
use std::{env, vec};

#[macro_use]
extern crate lazy_static;

lazy_static! {
    static ref PACKAGES: String = fs::read_to_string("data/packages.txt").unwrap();
    static ref PRODUCTS: Arc<Mutex<String>> = Arc::new(Mutex::new(
        fs::read_to_string("data/ideas-products.txt").expect("file not found")
    ));
    static ref CUSTOMERS: Arc<Mutex<String>> = Arc::new(Mutex::new(
        fs::read_to_string("data/ideas-customers.txt").expect("file not found")
    ));
}

struct Args {
    pub num_ideas: usize,
    pub num_idea_gen: usize,
    pub num_pkgs: usize,
    pub num_pkg_gen: usize,
    pub num_students: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    let num_ideas = args.get(1).map_or(Ok(80), |a| a.parse())?;
    let num_idea_gen = args.get(2).map_or(Ok(2), |a| a.parse())?;
    let num_pkgs = args.get(3).map_or(Ok(4000), |a| a.parse())?;
    let num_pkg_gen = args.get(4).map_or(Ok(6), |a| a.parse())?;
    let num_students = args.get(5).map_or(Ok(6), |a| a.parse())?;
    let args = Args {
        num_ideas,
        num_idea_gen,
        num_pkgs,
        num_pkg_gen,
        num_students,
    };

    hackathon(&args);
    Ok(())
}

fn per_thread_amount(thread_idx: usize, total: usize, threads: usize) -> usize {
    let per_thread = total / threads;
    let extras = total % threads;
    per_thread + (thread_idx < extras) as usize
}

fn hackathon(args: &Args) {
    // Use message-passing channel as event queue
    let (package_send, package_recv) = unbounded::<Event>();
    let (idea_send, idea_recv) = unbounded::<Event>();
    let mut threads = vec![];
    // Checksums of all the generated ideas and packages
    let mut idea_checksum = Arc::new(Mutex::new(Checksum::default()));
    let mut pkg_checksum = Arc::new(Mutex::new(Checksum::default()));
    // Checksums of the ideas and packages used by students to build ideas. Should match the
    // previous checksums.
    let mut student_idea_checksum = Arc::new(Mutex::new(Checksum::default()));
    let mut student_pkg_checksum = Arc::new(Mutex::new(Checksum::default()));

    let package_checksums: Vec<Checksum> = vec![];
    let package_checksums_prot = Arc::new(Mutex::new(package_checksums));

    let decoded_idea_checksums: HashMap<String, Checksum> = HashMap::new();
    let decoded_idea_checksum_protected = Arc::new(Mutex::new(decoded_idea_checksums));

    let packages_prot: Arc<Mutex<String>> = Arc::new(Mutex::new(PACKAGES.to_string()));

    // Spawn student threads
    for i in 0..args.num_students {
        let mut student = Student::new(
            i,
            Receiver::clone(&idea_recv),
            Receiver::clone(&package_recv),
        );
        let student_idea_checksum = Arc::clone(&student_idea_checksum);
        let student_pkg_checksum = Arc::clone(&student_pkg_checksum);
        let package_checksums_prot = Arc::clone(&package_checksums_prot);
        let decoded_idea_checksum_protected = Arc::clone(&decoded_idea_checksum_protected);
        let thread = spawn(move || {
            student.run(
                student_idea_checksum,
                student_pkg_checksum,
                decoded_idea_checksum_protected,
                package_checksums_prot,
            )
        });
        threads.push(thread);
    }

    // Spawn package downloader threads. Packages are distributed evenly across threads.
    let mut start_idx = 0;
    for i in 0..args.num_pkg_gen {
        let num_pkgs = per_thread_amount(i, args.num_pkgs, args.num_pkg_gen);
        let downloader = PackageDownloader::new(start_idx, num_pkgs, Sender::clone(&package_send));
        let pkg_checksum = Arc::clone(&pkg_checksum);
        let package_checksums_prot = Arc::clone(&package_checksums_prot);
        let package_names = Arc::clone(&packages_prot);
        start_idx += num_pkgs;

        let thread =
            spawn(move || downloader.run(pkg_checksum, package_checksums_prot, package_names));

        threads.push(thread);
    }
    assert_eq!(start_idx, args.num_pkgs);

    // Spawn idea generator threads. Ideas and packages are distributed evenly across threads. In
    // each thread, packages are distributed evenly across ideas.
    let mut start_idx = 0;
    for i in 0..args.num_idea_gen {
        let num_ideas = per_thread_amount(i, args.num_ideas, args.num_idea_gen);
        let num_pkgs = per_thread_amount(i, args.num_pkgs, args.num_idea_gen);
        let num_students = per_thread_amount(i, args.num_students, args.num_idea_gen);
        let decoded_idea_checksum_protected = Arc::clone(&decoded_idea_checksum_protected);
        let products = Arc::clone(&PRODUCTS);
        let customers = Arc::clone(&CUSTOMERS);
        let generator = IdeaGenerator::new(
            start_idx,
            num_ideas,
            num_students,
            num_pkgs,
            Sender::clone(&idea_send),
            products,
            customers,
        );

        let idea_checksum = Arc::clone(&idea_checksum);
        start_idx += num_ideas;

        let thread = spawn(move || generator.run(idea_checksum, decoded_idea_checksum_protected));
        threads.push(thread);
    }
    assert_eq!(start_idx, args.num_ideas);

    // Join all threads
    threads.into_iter().for_each(|t| t.join().unwrap());

    let idea = Arc::get_mut(&mut idea_checksum).unwrap().get_mut().unwrap();
    let student_idea = Arc::get_mut(&mut student_idea_checksum)
        .unwrap()
        .get_mut()
        .unwrap();
    let pkg = Arc::get_mut(&mut pkg_checksum).unwrap().get_mut().unwrap();
    let student_pkg = Arc::get_mut(&mut student_pkg_checksum)
        .unwrap()
        .get_mut()
        .unwrap();

    println!("Global checksums:\nIdea Generator: {}\nStudent Idea: {}\nPackage Downloader: {}\nStudent Package: {}", 
        idea, student_idea, pkg, student_pkg);
}
