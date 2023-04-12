// Copyright 2021 Twitter, Inc.
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0

//! A hashtable is used to lookup items and store per-item metadata.
//!
//! The [`HashTable`] design uses bulk chaining to reduce the per item overheads,
//! share metadata where possible, and provide better data locality.
//!
//! For a more detailed description of the implementation, please see:
//! <https://twitter.github.io/pelikan/2021/segcache.html>
//!
//! Our [`HashTable`] is composed of a base unit called a [`HashBucket`]. Each
//! bucket is a contiguous allocation that is sized to fit in a single
//! cacheline. This gives us room for a total of 8 64bit slots within the
//! bucket. The first slot of a bucket is used for per bucket metadata, leaving
//! us with up to 7 slots for items in the bucket:
//!
//! ```text
//!    ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
//!    │Bucket│ Item │ Item │ Item │ Item │ Item │ Item │ Item │
//!    │ Info │ Info │ Info │ Info │ Info │ Info │ Info │ Info │
//!    │      │      │      │      │      │      │      │      │
//!    │64 bit│64 bit│64 bit│64 bit│64 bit│64 bit│64 bit│64 bit│
//!    │      │      │      │      │      │      │      │      │
//!    └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
//! ```
//!
//! When a bucket is full, we may be able to chain another bucket from the
//! overflow area onto the primary bucket. To store a pointer to the next bucket
//! in the chain, we reduce the item capacity of the bucket and store the
//! pointer in the last slot. This can be repeated to chain additional buckets:
//!
//! ```text
//!    ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
//!    │Bucket│ Item │ Item │ Item │ Item │ Item │ Item │ Next │
//!    │ Info │ Info │ Info │ Info │ Info │ Info │ Info │Bucket│
//!    │      │      │      │      │      │      │      │      │──┐
//!    │64 bit│64 bit│64 bit│64 bit│64 bit│64 bit│64 bit│64 bit│  │
//!    │      │      │      │      │      │      │      │      │  │
//!    └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘  │
//!                                                               │
//! ┌─────────────────────────────────────────────────────────────┘
//! │
//! │  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
//! │  │ Item │ Item │ Item │ Item │ Item │ Item │ Item │ Next │
//! │  │ Info │ Info │ Info │ Info │ Info │ Info │ Info │Bucket│
//! └─▶│      │      │      │      │      │      │      │      │──┐
//!    │64 bit│64 bit│64 bit│64 bit│64 bit│64 bit│64 bit│64 bit│  │
//!    │      │      │      │      │      │      │      │      │  │
//!    └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘  │
//!                                                               │
//! ┌─────────────────────────────────────────────────────────────┘
//! │
//! │  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
//! │  │ Item │ Item │ Item │ Item │ Item │ Item │ Item │ Item │
//! │  │ Info │ Info │ Info │ Info │ Info │ Info │ Info │ Info │
//! └─▶│      │      │      │      │      │      │      │      │
//!    │64 bit│64 bit│64 bit│64 bit│64 bit│64 bit│64 bit│64 bit│
//!    │      │      │      │      │      │      │      │      │
//!    └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
//! ```
//!
//! This works out so that we have capacity to store 7 items for every bucket
//! allocated to a chain.
//!

// hashtable

/// The number of slots within each bucket
const N_BUCKET_SLOT: usize = 8;

/// Maximum number of buckets in a chain. Must be <= 255.
const MAX_CHAIN_LEN: u64 = 16;

use crate::*;
use crate::datapool::*;
use ahash::RandomState;
use ::rand::thread_rng;
use core::num::NonZeroU32;

use rustcommon_time::CoarseInstant as Instant;

mod hash_bucket;

pub(crate) use hash_bucket::*;

#[derive(Debug)]
struct IterState {
    start: bool,
    bucket_id: usize,
    buckets_len: usize,
    item_slot: usize,
    chain_len: usize,
    chain_idx: usize,
    finished: bool,
}

impl IterState {
    fn new(hashtable: &mut HashTable, hash: u64) -> Self {
        let bucket_id = (hash & hashtable.mask) as usize;
        let buckets_len = hashtable.data.len();
        let bucket = hashtable.data[bucket_id];
        let chain_len = chain_len(bucket.data[0]) as usize;

        Self {
            start: true,
            bucket_id,
            buckets_len,
            // we start with item_slot 0 because in Flash there's no metadata
            item_slot: 0,
            chain_len,
            chain_idx: 0,
            finished: false,
        }
    }

    fn n_item_slot(&self) -> usize {
        // if this is the last bucket in the chain, the final slot contains item
        // info entry, otherwise it points to the next bucket and should not be
        // treated as an item slot
        if self.chain_idx == self.chain_len {
            N_BUCKET_SLOT
        } else {
            N_BUCKET_SLOT - 1
        }
    }
}

struct IterMut<'a> {
    ptr: *mut PrimaryHashBucket,
    hashtable: &'a mut HashTable,
    state: IterState,
}

impl<'a> IterMut<'a> {
    fn new(hashtable: &'a mut HashTable, hash: u64) -> Self {
        let state = IterState::new(hashtable, hash);

        // ptr to the DRAM part to optimise bucket fetching
        let ptr = hashtable.data.as_mut_ptr();

        Self {
            ptr,
            hashtable,
            state,
        }
    }
}

impl<'a> Iterator for IterMut<'a> {
    type Item = &'a mut u64;

    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        #[cfg(feature = "htbl_usage")]
        println!("Bucket Id:{}; Item slot:{}", self.state.bucket_id, self.state.item_slot);

        if self.state.finished {
            return None;
        }

        let n_item_slot = self.state.n_item_slot();

        // get DRAM data
        if self.state.start {
            self.state.start = false;
            let item_info =
                unsafe { &mut (*self.ptr.add(self.state.bucket_id)).data[1] };
            return Some(item_info);
        }

        // SAFETY: this assert ensures memory safety for the pointer operations
        // that follow as in-line unsafe blocks. We first check to make sure the
        // bucket_id is within range for the slice of buckets. As long as this
        // holds true, the pointer operations are safe.
        assert!(
            self.state.bucket_id < self.state.buckets_len,
            "bucket id not in range"
        );

        // get Flash data
        let ptr: &mut [u64] = unsafe {
            std::mem::transmute(self.hashtable.data_flash.as_mut_slice())
        };
       let item_info = &mut ptr[self.state.bucket_id * 8 + self.state.item_slot]; 

        // update iter state
        if self.state.item_slot < n_item_slot - 1 {
            self.state.item_slot += 1;
        } else {
            // finished iterating in this bucket, see if it's chained
            self.state.finished = true;
        }

        Some(item_info)
    }
}

/// Main structure for performing item lookup. Contains a contiguous allocation
/// of [`PrimaryHashBucket`]s which are used to store first-item info and metadata.
/// 128 bytes
#[repr(C)]
pub(crate) struct HashTable {
    hash_builder: Box<RandomState>,
    power: u64,
    mask: u64,
    data: Box<[PrimaryHashBucket]>,
    started: CoarseInstant,
    next_to_chain: u64,
    data_flash: Box<dyn Datapool>,
    _pad : [u8; 56],
}

impl HashTable {
    /// Creates a new hashtable with a specified power and overflow factor.
    pub fn new(power: u8, overflow_factor: f64) -> HashTable {
        if overflow_factor < 0.0 {
            fatal!("hashtable overflow factor must be >= 0.0");
        }

        // overflow factor is effectively bounded by the max chain length
        if overflow_factor > MAX_CHAIN_LEN as f64 {
            fatal!("hashtable overflow factor must be <= {}", MAX_CHAIN_LEN);
        }

        // # 2^power slots that can fit in DRAM with mem=2^(power+3) bytes 
        let slots = 1_u64 << power;
        let buckets = slots >> 3;
        let mask = buckets - 1;

        // Allcoate in DRAM
        let mut data = Vec::with_capacity(0);
        data.reserve_exact(buckets as usize);
        data.resize(buckets as usize, PrimaryHashBucket::new());
        let data_size = buckets as usize * primary_bucket_size();
        debug!("DRAM hashtable size in bytes {}", data_size);

        // Allocate in Flash
        let total_buckets_flash = (buckets as f64 * (1.0 + overflow_factor)).ceil() as usize;
        let data_size_flash = total_buckets_flash * flash_bucket_size();
        let data_flash: Box<dyn Datapool> = Box::new(Memory::create(data_size_flash, true));

        debug!("Flash hashtable size in bytes {}", data_size_flash);
        info!(
            "hashtable has: {} primary slots across {} primary buckets and {} total buckets",
            slots, buckets, total_buckets_flash,
        );
        info!("Hashtable power: {}; Hashbuckets: {}", power, buckets);

        let hash_builder = RandomState::with_seeds(
            0xbb8c484891ec6c86,
            0x0522a25ae9c769f9,
            0xeed2797b9571bc75,
            0x4feb29c1fbbd59d0,
        );

        Self {
            hash_builder: Box::new(hash_builder),
            power: power.into(),
            mask,
            data: data.into_boxed_slice(),
            started: Instant::recent(),
            next_to_chain: buckets as u64,
            data_flash: data_flash,
            _pad : [0; 56],
        }
    }

    /// Get mutable hash bucket by the id
    pub fn get_bucket_flash(&mut self, bucket_id: usize) -> HashBucket {
        let begin = flash_bucket_size() * bucket_id;
        let end = flash_bucket_size() * (bucket_id + 1);
        let hash_bucket = HashBucket::from_raw_parts(
            &mut self.data_flash.as_mut_slice()[begin..end],
        );
        hash_bucket
    }

    pub fn get_next_bucket_id(&self, bucket_id: usize) -> usize {
        let end = flash_bucket_size() * (bucket_id + 1);
        let x = self.data_flash.as_slice()[end - 1] as usize;
        x
    }

    /// Lookup an item by key and return it
    pub fn get(&mut self, key: &[u8], segments: &mut Segments) -> Option<Item> {
        let hash = self.hash(key);
        let tag = tag_from_hash(hash);
        let bucket_id = hash & self.mask;

        let bucket_info = self.data[bucket_id as usize].data[0];
        let curr_ts = (Instant::recent() - self.started).as_secs() as u64 & PROC_TS_MASK;

        // update access time in Bucket metadata
        if curr_ts != get_ts(bucket_info) {
            self.data[bucket_id as usize].data[0] = (bucket_info & !TS_MASK) | (curr_ts as u64);

            let iter = IterMut::new(self, hash);
            for item_info in iter {
                *item_info &= CLEAR_FREQ_SMOOTH_MASK;
            }
        }

        let tag_exists = self.check_reduced_tag(key);

        let iter = IterMut::new(self, hash);
        for (i, current_info) in iter.enumerate() {
            if i > 0 && tag_exists.is_none() {
                return None;
            } else if let Some(idx) = tag_exists {
                if idx < i {
                    continue;
                }
            }

            if get_tag(*current_info) == tag {
                let current_item = segments.get_item(*current_info).unwrap();
                if current_item.key() == key {
                    // update item frequency
                    let mut freq = get_freq(*current_info);
                    if freq < 127 {
                        let rand = thread_rng().gen::<u64>();
                        if freq <= 16 || rand % freq == 0 {
                            freq = ((freq + 1) | 0x80) << FREQ_BIT_SHIFT;
                        } else {
                            freq = (freq | 0x80) << FREQ_BIT_SHIFT;
                        }
                        *current_info = (*current_info & !FREQ_MASK) | freq;
                    }

                    let item = Item::new(
                        current_item,
                        get_cas(self.data[(hash & self.mask) as usize].data[0]),
                    );
                    item.check_magic();

                    return Some(item);
                }
            }
        }

        None
    }

    /// Lookup an item by key and return it without incrementing the item
    /// frequency. This may be used to compose higher-level functions which do
    /// not want a successful item lookup to count as a hit for that item.
    pub fn get_no_freq_incr(&mut self, key: &[u8], segments: &mut Segments) -> Option<Item> {
        let hash = self.hash(key);
        let tag = tag_from_hash(hash);
        let iter = IterMut::new(self, hash);

        for current_info in iter {
            if get_tag(*current_info) == tag {
                let current_item = segments.get_item(*current_info).unwrap();
                if current_item.key() == key {
                    let item = Item::new(
                        current_item,
                        get_cas(self.data[(hash & self.mask) as usize].data[0]),
                    );
                    item.check_magic();

                    return Some(item);
                }
            }
        }

        None
    }

    /// Return the frequency for the item with the key
    pub fn get_freq(&mut self, key: &[u8], segment: &mut Segment, offset: u64) -> Option<u64> {
        let hash = self.hash(key);
        let tag = tag_from_hash(hash);

        let iter = IterMut::new(self, hash);

        for item_info in iter {
            if get_tag(*item_info) == tag
                && get_seg_id(*item_info) == Some(segment.id())
                && get_offset(*item_info) == offset
            {
                return Some(get_freq(*item_info) & 0x7F);
            }
        }

        None
    }

    /// Relinks the item to a new location
    #[allow(clippy::result_unit_err)]
    pub fn relink_item(
        &mut self,
        key: &[u8],
        old_seg: NonZeroU32,
        new_seg: NonZeroU32,
        old_offset: u64,
        new_offset: u64,
    ) -> Result<(), ()> {
        let hash = self.hash(key);
        let tag = tag_from_hash(hash);

        let iter = IterMut::new(self, hash);

        for current_info in iter {
            if get_tag(*current_info) == tag {
                if get_seg_id(*current_info) == Some(old_seg) && get_offset(*current_info) == old_offset {
                    *current_info = build_item_info(tag, new_seg, new_offset);
                    return Ok(());
                }
            }
        }

        Err(())
    }

    /// Inserts a new item into the hashtable. This may fail if the hashtable is
    /// full.
    #[allow(clippy::result_unit_err)]
    pub fn insert(
        &mut self,
        item: RawItem,
        seg: NonZeroU32,
        offset: u64,
        ttl_buckets: &mut TtlBuckets,
        segments: &mut Segments,
    ) -> Result<(), ()> {

        let hash = self.hash(item.key());
        let tag = tag_from_hash(hash);
        let bucket_id = (hash & self.mask) as usize;

        // check the item magic
        item.check_magic();

        let mut insert_item_info = build_item_info(tag, seg, offset);

        let mut removed: Option<u64> = None;

        let iter = IterMut::new(self, hash);

        // iter the current PrimaryHashBucket to see if there's blank slot
        for (i, item_info) in iter.enumerate() {
            if get_tag(*item_info) != tag {
                if insert_item_info != 0 && *item_info == 0 {
                    // found a blank slot
                    *item_info = insert_item_info;
                    insert_item_info = 0;

                    if i > 0 {
                        let x = &mut self.data[bucket_id].data[2];
                        // clear the original tag
                        *x &= !(0xff << (i-1)*8);
                        // add current tag
                        *x |= ((tag >> 52) & 0xff) << (i-1)*8;
                    }

                    break;
                } else {
                    // item already exists
                    continue;
                }
            } else {
                if segments.get_item(*item_info).unwrap().key() == item.key() {
                    // update existing key
                    removed = Some(*item_info);
                    *item_info = insert_item_info;
                    insert_item_info = 0;

                    if i > 0 {
                        let x = &mut self.data[bucket_id].data[2];
                        // clear the original tag
                        *x &= !(0xff << (i-1)*8);
                        // add current tag
                        *x |= ((tag >> 52) & 0xff) << (i-1)*8;
                    }
                    break;
                }
            }
        }

        if let Some(removed_item) = removed {
            let _ = segments.remove_item(removed_item, true, ttl_buckets, self);
        }

        // bulk-chaining of hashtable happens here
        // item hasn't been inserted yet, since there's no empty slot/repeated keys in the current HashBucket,
        // so need to bulk-chain with another (empty) HashBucket
        if insert_item_info != 0 {
            let mut bucket_id = (hash & self.mask) as usize;
            let chain_len = chain_len(self.data[bucket_id].data[0]);

            if chain_len < MAX_CHAIN_LEN && (self.next_to_chain as usize) < self.data.len() {
                // we need to chase through the buckets to get the id of the last
                // bucket in the chain
                for _ in 0..chain_len {
                    bucket_id = self.get_next_bucket_id(bucket_id);
                }

                // chain (next_to_chain)th HashBucket from the Flash section
                let next_id = self.next_to_chain as usize;
                self.next_to_chain += 1;

                self.get_bucket_flash(next_id).data[0] = self.get_next_bucket_id(bucket_id) as u64;

                let current_bucket = self.get_bucket_flash(bucket_id);
                current_bucket.data[1] = insert_item_info;

                insert_item_info = 0;
                current_bucket.data[N_BUCKET_SLOT - 1] = next_id as u64;

                self.data[(hash & self.mask) as usize].data[0] += 0x0001_0000_0000_0000;
            }
        }

        // update HashBucket metadata
        if insert_item_info == 0 {
            self.data[(hash & self.mask) as usize].data[0] += 1;
            Ok(())
        } else {
            Err(())
        }
    }

    /// Used to implement higher-level CAS operations. This function looks up an
    /// item by key and checks if the CAS value matches the provided value.
    ///
    /// A success indicates that the item was found with the CAS value provided
    /// and that the CAS value has now been updated to a new value.
    ///
    /// A failure indicates that the CAS value did not match or there was no
    /// matching item for that key.
    pub fn try_update_cas<'a>(
        &mut self,
        key: &'a [u8],
        cas: u32,
        segments: &mut Segments,
    ) -> Result<(), SegError<'a>> {
        let hash = self.hash(key);
        let tag = tag_from_hash(hash);
        let bucket_id = hash & self.mask;

        let iter = IterMut::new(self, hash);

        for item_info in iter {
            if get_tag(*item_info) == tag {
                let item = segments.get_item(*item_info).unwrap();
                if item.key() == key {
                    // update item frequency
                    let mut freq = get_freq(*item_info);
                    if freq < 127 {
                        let rand = thread_rng().gen::<u64>();
                        if freq <= 16 || rand % freq == 0 {
                            freq = ((freq + 1) | 0x80) << FREQ_BIT_SHIFT;
                        } else {
                            freq = (freq | 0x80) << FREQ_BIT_SHIFT;
                        }
                        *item_info = (*item_info & !FREQ_MASK) | freq;
                    }

                    if cas == get_cas(self.data[bucket_id as usize].data[0])  {
                        self.data[bucket_id as usize].data[0] += 1;
                        return Ok(());
                    } else {
                        return Err(SegError::Exists);
                    }
                }
            }
        }

        Err(SegError::NotFound)
    }

    /// Removes the item with the given key
    pub fn delete(
        &mut self,
        key: &[u8],
        ttl_buckets: &mut TtlBuckets,
        segments: &mut Segments,
    ) -> bool {
        let hash = self.hash(key);
        let tag = tag_from_hash(hash);

        let iter = IterMut::new(self, hash);

        let mut removed: Option<u64> = None;

        for item_info in iter {
            if get_tag(*item_info) == tag {
                let item = segments.get_item(*item_info).unwrap();
                if item.key() != key {
                    continue;
                } else {
                    removed = Some(*item_info);
                    *item_info = 0;
                    break;
                }
            }
        }

        if let Some(removed_item) = removed {
            let _ = segments.remove_item(removed_item, false, ttl_buckets, self);
            true
        } else {
            false
        }
    }

    /// Evict a single item from the cache
    pub fn evict(&mut self, key: &[u8], offset: i32, segment: &mut Segment) -> bool {
        let result = self.remove_from(key, offset, segment);
        result
    }

    /// Expire a single item from the cache
    pub fn expire(&mut self, key: &[u8], offset: i32, segment: &mut Segment) -> bool {
        let result = self.remove_from(key, offset, segment);
        result
    }

    /// Internal function that removes an item from a segment
    fn remove_from(&mut self, key: &[u8], offset: i32, segment: &mut Segment) -> bool {
        let hash = self.hash(key);
        let tag = tag_from_hash(hash);
        let evict_item_info = build_item_info(tag, segment.id(), offset as u64);

        let iter = IterMut::new(self, hash);

        for (i, item_info) in iter.enumerate() {
            let current_item_info = clear_freq(*item_info);
            if get_tag(current_item_info) != tag {
                continue;
            }

            if get_seg_id(current_item_info) != Some(segment.id())
                || get_offset(current_item_info) != offset as u64
            {
                continue;
            }

            if evict_item_info == current_item_info {
                segment.remove_item(current_item_info, false);
                *item_info = 0;

                if i > 0 {
                    let bucket_id = (hash & self.mask) as usize;
                    self.data[bucket_id].data[2] &= !(0xff << (i-1)*8);
                }
                return true;
            }
        }

        false
    }

    /// Internal function used to calculate a hash value for a key
    fn hash(&self, key: &[u8]) -> u64 {
        let mut hasher = self.hash_builder.build_hasher();
        hasher.write(key);
        hasher.finish()
    }

    /// Internal function used to check if tag from hash(key) corresponds to
    /// one of the tags in metadata
    fn check_reduced_tag(&self, key: &[u8]) -> Option<usize> {
        // get the shifted tag
        let hash = self.hash(key);
        let tag = tag_from_hash(hash) >> 52;
        
        // get the metadata tags
        let tags = self.data[(hash & self.mask) as usize].data[2];
        let mask_tmp: u64 = 0xff;

        // MY-TODO: extract reduced tag length to configs
        for i in 0..8 {
            if (tag & mask_tmp) == ((tags >> (i*8)) & mask_tmp) {
                return Some(i);
            }
        }
        None
    }
}
