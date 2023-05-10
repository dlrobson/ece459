use std::collections::HashMap;
use std::env;
use std::sync::{Arc, Mutex};
use std::thread::spawn;

// The character 'a' is represented by ASCII code 97. If you want to treat
// 'a' as index 0 of the alphabet, the offset here helps.
const ASCII_OFFSET: usize = 97;
const MAX_THREADS: usize = 4;
const NUM_LETTERS: usize = 26;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        println!("Usage: {} <filename>", args[0]);
        return;
    }
    let filename = &args[1];

    let words = read_words_from_file(filename);
    println!("There are {} words to evaluate.", words.len());

    let letter_frequency = letter_frequency(&words);
    let letter_frequency = rank(letter_frequency);

    let word_score = score_words(&words, letter_frequency);

    let max = find_max_score(&word_score);
    println!("Max score is: {}.", max);
    let max_words = find_max_words(word_score, &max);

    print_suggestions(max_words);
    println!();
}

fn letter_frequency(words: &[String]) -> Vec<i32> {
    let mut letter_frequencies: Vec<i32> = vec![0; NUM_LETTERS];
    let mut letter_frequencies_prot = Arc::new(Mutex::new(letter_frequencies));

    let mut threads = vec![];
    
    let mut start_i = 0;
    let mut end_i = 0;

    for i in 0..MAX_THREADS {
        // Determine indices
        start_i = words.len() / MAX_THREADS * i;
        if i + 1 != MAX_THREADS {
            end_i = words.len() / MAX_THREADS * (i + 1);
        }
        else {
            end_i = words.len();
        }
        
        let letter_frequencies_prot = Arc::clone(&letter_frequencies_prot);
        let words_vec = words.clone().to_vec();
        let thread = spawn(move || {
            calculate_frequencies(
                letter_frequencies_prot,
                words_vec,
                start_i.clone(),
                end_i.clone(),
            )
        });
        threads.push(thread);
    }

    threads.into_iter().for_each(|t| t.join().unwrap());

    Arc::get_mut(&mut letter_frequencies_prot)
        .unwrap()
        .get_mut()
        .unwrap()
        .to_vec()
}

fn calculate_frequencies(letter_frequencies_prot: Arc<Mutex<Vec<i32>>>,
                         words: Vec<String>,
                         start_i: usize,
                         end_i: usize) {
    let mut letter_frequencies_copy: Vec<i32> = vec![0; NUM_LETTERS];
    for word_i in start_i..end_i {
        let word: &String = &words[word_i];
        for c in word.chars() {
            let index = c as usize - ASCII_OFFSET;
            letter_frequencies_copy[index] += 1;
        }
    }
    {
        let mut letter_frequencies = letter_frequencies_prot.lock().unwrap();
        for i in 0..NUM_LETTERS {
            letter_frequencies[i] += letter_frequencies_copy[i];
        }
    }
}

fn score_words(words: &[String], frequency: Vec<i32>) -> HashMap<String, i32> {

    let mut scores: HashMap<String, i32> = HashMap::new();
    let mut scores_prot = Arc::new(Mutex::new(scores));
    let mut threads = vec![];
    
    let mut start_i = 0;
    let mut end_i = 0;

    for i in 0..MAX_THREADS {
        // Determine indices
        start_i = words.len() / MAX_THREADS * i;
        if i + 1 != MAX_THREADS {
            end_i = words.len() / MAX_THREADS * (i + 1);
        }
        else {
            end_i = words.len();
        }
        
        let scores_prot = Arc::clone(&scores_prot);
        let words_vec = words.clone().to_vec();
        let frequency_copy = frequency.clone();
        let thread = spawn(move || {
            calculate_score(
                frequency_copy,
                scores_prot,
                words_vec,
                start_i.clone(),
                end_i.clone(),
            )
        });
        threads.push(thread);
    }

    threads.into_iter().for_each(|t| t.join().unwrap());

    Arc::get_mut(&mut scores_prot)
        .unwrap()
        .get_mut()
        .unwrap()
        .clone()
}

fn calculate_score(letter_frequencies: Vec<i32>,
                         scores_prot: Arc<Mutex<HashMap<String, i32>>>,
                         words: Vec<String>,
                         start_i: usize,
                         end_i: usize) {
    
    let mut scores_subset: HashMap<String, i32> = HashMap::new();
    for word_i in start_i..end_i {
        let word: &String = &words[word_i];
        scores_subset.insert(word.to_string(), 0);
        for (i, c) in word.chars().enumerate() {
            let letters_so_far = &word[0..i];
            if letters_so_far.contains(c) {
                continue;
            }
            let index = c as usize - ASCII_OFFSET;
            *scores_subset.get_mut(word).unwrap() += letter_frequencies[index];
        }
    }

    {
        let mut scores = scores_prot.lock().unwrap();
        scores.extend(scores_subset);
    }
}

fn find_max_score(word_score: &HashMap<String, i32>) -> i32 {
    let mut max = 0;
    for (_word, score) in word_score.iter() {
        if *score > max {
            max = *score;
        }
    }
    max
}

fn find_max_words(word_score: HashMap<String, i32>, max: &i32) -> Vec<String> {
    let mut max_words = Vec::new();
    for (word, score) in word_score.iter() {
        if *score == *max {
            max_words.push(word.clone());
        }
    }
    max_words
}

fn print_suggestions(max_words: Vec<String>) {
    print!("Suggestion(s): ");
    let mut first = true;
    for item in max_words {
        if !first {
            print!(", ");
        }
        print!("{}", item);
        first = false;
    }
}

fn read_words_from_file(inputfilename: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut rdr = csv::Reader::from_path(inputfilename).unwrap();
    for line in rdr.records() {
        let word = String::from(line.unwrap().get(0).unwrap());
        if word.len() > 5 {
            panic!("Word too long: {}", word);
        }
        words.push(word);
    }
    words
}

fn rank(frequency: Vec<i32>) -> Vec<i32> {
    let mut ranks = frequency.clone();
    ranks.sort_unstable();
    let mut map: HashMap<i32, i32> = HashMap::new();
    let mut index = 1;
    let mut prev = *ranks.get(0).unwrap();
    map.insert(prev, index);

    for i in 1..frequency.len() {
        let cur = ranks.get(i).unwrap();
        if prev != *cur {
            index += 1;
        }
        map.insert(*cur, index);
        prev = *cur
    }

    for j in 0..frequency.len() {
        ranks[j] = *map.get(&frequency[j]).unwrap();
    }
    ranks
}