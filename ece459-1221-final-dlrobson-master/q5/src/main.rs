use std::fs::File;
use std::io::{self, Read as _};
use std::time::Instant;

use json;
use simsearch::SimSearch;

fn main() -> io::Result<()> {
    let mut engine = SimSearch::new();

    let mut file = File::open("./pg.json")?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;

    let j = json::parse(&content).unwrap();

    for title in j.members() {
        engine.insert(title.as_str().unwrap(), title.as_str().unwrap());
    }

    let pattern = "Sir Robert de Umfraville (c. 1363 – 1437) was a late medieval English knight who took part in the later stages of the Hundred Years' War, particularly against Scotland.";

    let start = Instant::now();
    let res = engine.search(&pattern);
    let end = Instant::now();

    println!("pattern: {:?}", pattern.trim());
    let start1 = Instant::now();
    println!("results: {:?}", res);
    let end1 = Instant::now();
    println!("time: {:?}", end - start);
    println!("printtime: {:?}", end1 - start1);
    Ok(())
}
