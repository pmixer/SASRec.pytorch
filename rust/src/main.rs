use std::fs::File;
use std::path::Path;
use std::io::{self, BufRead};

use std::collections::HashMap;

pub struct Dataset {
    pub sequences: HashMap<usize, Vec<usize>>,
}

impl Dataset {
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = io::BufReader::new(file);

        let mut sequences = HashMap::new();

        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let user: usize = parts[0].parse().unwrap_or(0);
                let item: usize = parts[1].parse().unwrap_or(0);

                sequences.entry(user).or_insert_with(Vec::new).push(item);
            }
        }

        Ok(Self {sequences})
    }

    pub fn get_sequences(&self) {
        // return vec of vec(better in csr or other formats?) on host
    }
    // then sequences_to_ragged_tensor(most recent item aligned) on device

    pub fn print_sequences(&self) {
        let mut max_len : usize = 0;
        let mut min_len : usize = std::usize::MAX;
        let mut all_len : usize = 0;
        for (user, items) in &self.sequences {
            println!("{}->", user);
            all_len += items.len();
            max_len = max_len.max(items.len());
            min_len = min_len.min(items.len());
            for item in items {
                print!("{}, ", item);
            }
            println!("");
        }
        println!("n_sequences: {}", self.sequences.len());
        println!("min length: {}", min_len);
        println!("avg length: {}", all_len / self.sequences.len());
        println!("max length: {}", max_len);
    }
}



fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = Dataset::new("../python/data/ml-1m.txt").expect("Failed to load the dataset");
    dataset.print_sequences();
    Ok(())
}
