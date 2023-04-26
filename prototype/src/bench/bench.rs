
use super::reader::{Reader, ReaderError};
use super::request::{Request, Op}; 
use super::cache::Cache; 
use log::{info, debug}; 

#[derive(Debug)]
pub struct Bench {
    pub reader: Reader,
    pub cache: Cache, 
    pub n_get: u64,
    pub n_set: u64,
    pub n_del: u64,
    pub n_get_miss: u64,

    pub n_get_interval: u64,
    pub n_set_interval: u64,
    pub n_del_interval: u64,
    pub n_get_miss_interval: u64,

    pub bench_time: i32, 
    pub warmup_sec: i32,

    pub start_time: std::time::Instant, 
    pub end_time: std::time::Instant, 
    pub trace_time: i32, 
    pub report_interval: i32, // in seconds
}

impl Bench {
    pub fn new(reader: Reader, cache: Cache, bench_time: i32, warmup_sec: i32, report_interval: i32) -> Bench {
        info!("{} {} {}",
            cache.get_name(), cache.get_size_in_mb(), 
            reader.trace_path.split("/").last().unwrap()
        );
        Bench {
            reader: reader,
            cache: cache,
            n_get: 0,
            n_set: 0,
            n_del: 0,
            n_get_miss: 0,

            n_get_interval: 0,
            n_set_interval: 0,
            n_del_interval: 0,
            n_get_miss_interval: 0,
            
            bench_time: bench_time,
            warmup_sec: warmup_sec, 

            start_time: std::time::Instant::now(), 
            end_time: std::time::Instant::now(),
            trace_time: 0,
            report_interval: report_interval,
        }
    }
    
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.n_get = 0;
        self.n_set = 0;
        self.n_get_miss = 0;
        self.n_del = 0;
    }
    
    pub fn report(& mut self) {
        // self.runtime = std::time::SystemTime::now().duration_since(self.start_time).unwrap().as_secs() as f64;
        self.n_get += self.n_get_interval; 
        self.n_set += self.n_set_interval;
        self.n_del += self.n_del_interval;
        self.n_get_miss += self.n_get_miss_interval;

        let runtime = self.start_time.elapsed().as_secs_f64();
        let n_req = self.n_get + self.n_set + self.n_del;
        let trace_time_str = format!("{:.2}", self.trace_time as f64 / 3600.0);

        println!("{} req, trace {} hour, {:.2} sec, throughput {:.2} MQPS, miss ratio {:.4}", 
            self.n_get, trace_time_str, runtime, n_req as f64 / runtime / 1e6, 
            self.n_get_miss as f64 / (self.n_get as f64 + self.n_get_miss as f64), 
        );

        self.n_get_interval = 0;
        self.n_set_interval = 0;
        self.n_del_interval = 0;
        self.n_get_miss_interval = 0;
    }

    pub fn run(&mut self) {
        let mut get_time = 0;
        let mut set_time = 0;
        let mut del_time = 0;

        self.start_time = std::time::Instant::now();
        let mut request: Request = Request::default(); 
        if let Err(err) = self.reader.read(&mut request) {
            eprintln!("cannot read trace {:?}", err);
        } else {
            debug!("first request {}", request); 
        }

        let trace_start = request.real_time as i32;
        let mut next_report_interval: i32 = if self.report_interval > 0 {
            trace_start + self.report_interval
        } else {
            i32::MAX
        };
        
        let mut has_warmup = false; 
        let mut buf: Vec<u8> = Vec::with_capacity(1024*1024*8);
        buf.resize(1024*1024*8, 0);

        loop {
            if !has_warmup && request.real_time as i32 - trace_start > self.warmup_sec {
                self.n_get = 0;
                self.n_set = 0;
                self.n_get_miss = 0;
                self.n_del = 0;

                self.start_time = std::time::Instant::now();
                has_warmup = true;
                info!("{:.2} hr warmup done", (request.real_time as i32 - trace_start) as f64 / 3600.0);
            }
            match self.reader.read(&mut request) { 
                Ok(()) => {
                    match request.op {
                        Op::Get => {
                            let now = std::time::Instant::now();
                            let ret = self.cache.get(&request, &mut buf);
                            if !ret {
                                let now = std::time::Instant::now();
                                self.n_get_miss_interval += 1;
                                self.n_set_interval += 1;
                                self.cache.set(&request);
                                set_time += now.elapsed().as_micros();
                            } else {
                                get_time += now.elapsed().as_micros();
                                self.n_get_interval += 1;
                            }
                        }
                        Op::Set => {
                            let now = std::time::Instant::now();
                            self.n_set_interval += 1;
                            self.cache.set(&request); 
                            set_time += now.elapsed().as_micros();
                        }
                        Op::Del => {
                            let now = std::time::Instant::now();
                            self.n_del_interval += 1;
                            if ! self.cache.del(&request) {
                            }
                            del_time += now.elapsed().as_micros();
                        }
                        Op::Invalid => {
                            panic!("invalid op");
                        }
                    }
                }, 
                Err(e) => { 
                    match e {
                        ReaderError::EOF => {
                            break;
                        }
                        ReaderError::SkipReq => {
                            // println!("skip request {}", request); 
                        }
                        _ => {
                            eprintln!("error in reading trace {:?}", e);
                            break;
                        }
                    }
                }, 
            } 

            if request.real_time as i32 > next_report_interval {
                while next_report_interval < request.real_time as i32 {
                    next_report_interval += self.report_interval;
                }
                self.trace_time = request.real_time as i32 - trace_start;
                self.report();
            } 

            if request.real_time as i32 > self.bench_time + 1 {
                break;
            }
        }

        info!("GET {} us, SET {} us, DEL {} us", get_time, set_time, del_time);
        info!("GET# {}, SET# {}, DEL# {}", self.n_get, self.n_set, self.n_del);
        self.trace_time = request.real_time as i32 - trace_start;
        self.end_time = std::time::Instant::now();
    }
}


