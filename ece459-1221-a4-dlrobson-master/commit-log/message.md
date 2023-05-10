# Removed Redundancy, Reduced Lock Time, and Minimized Channel Activity

## Summary
<!-- A one-paragraph high-level description of the work done. -->

Throughout the code, there were several inefficiencies. This included redundant calculations that were done previously, overlocking of mutexes causing lots of blocks throughout the threads, additional channel simplify logic, and expensive operations that could be simplified. In general, holding locks for a minimum amount of time is necessary for a efficient program. And to go along with this, the actions that are required to be completed with the lock should be as efficient as possible to reduce the lock time.

## Technical details
<!-- Anything interesting or noteworthy for the reviewer about how the work is done (3-5 paragraphs) -->

In `checksums.rs`, checksums are now kept as decoded, to avoid the unnecessary encode. The idea and package checksums as they're decoded are held in a vector/hashmap for easier future reference. The Xor Function was modified to be in place, removing an unnecessary copy.

In `idea.rs`, idea checksums are held in a hashmap to avoid recalculating the checksums. They are referenced by the idea name string. The events are passed to a specific idea channel to avoid parsing for both ideas and packages.

The significant update in `main.rs` is that the 3 files are now read a single time, and are passed to their corresponding structs.

There were several inefficiencies in `package.rs`. To start, nth iterator is an O(n) operation, and was replaced with a simple modulus calculation, significantly decreasing execution time. The packages checksums are stored in a vector corresponding to the index of the package, to avoid recalculating the checksums. The packages checksum are stored and updated at the end of the `run()` function, to avoid constantly attempting to lock the `pkg_checksum` mutex to update the global checksum. A separate channel was created specifically for the packages channel, to avoid the student.run to search for both ideas and packages within the same channel.

Several changes were made to `student.rs`. The update of the global idea and package checksum is now done sequentially together to avoid consistent attempts to lock the mutex. This fills a vector, which empties when we fail to obtain an event, or when we've run out of ideas. There are now 2 separate channels: one for ideas and one for packages. The student does not check the idea's channel if it currently has an idea. It then attempts to find a package. If in either case it fails to obtain an event, it flushes out the vector of checksums. This gives the channels oppurtunity to refill with packages/ideas. Also, `stdout.lock()` after the string has been assembled. This minimizes the time that the lock is held.

## Testing for correctness
<!-- Something about how you tested this code for correctness and know it works (about 1 paragraph). -->

The final global checksums were confirmed to have not changed before any changes. The intermediate checksums were affected, however this was allowed in the lab manual. Also, different intermediate outcomes are consistently expected. Also, valgrind did not list any other memory leaks than there were originally.

## Testing for performance.
<!-- Something about how you tested this code for performance and know it is faster (about 3 paragraphs, referring to the flamegraph as appropriate.) -->

The new optimized code is significantly more efficient. The flamegraph helps demonstrate this. The majority of the computation time is done outside of main, where the code is initialized. From my understanding this is not easily optimizable, and is outside of the scope of this project. The remaining computation time is from writing to `stdout()`.

The original flamegraph spent significant time using the `nth` iterator, reading files, as well as sending and receiving channels. The time now spent blocking for receives or executing a send is now 4%, compared to the original 40%. The attached flamegraph is the final result of the optimized code on `ecetesla0`.

Some results from various machines are displayed below from hyperfine, on various machines. These tests were taken at 11pm which is likely peak ECE lab activity time, so results may vary.


| num_ideas, num_packages    | ecetesla0      | ecetesla1      | i7-8650u       |
| -------------------------- | -------------- | -------------- | -------------- |
| 80, 4000 (original impl)   | 478.6 ms       | 53.0 ms        | 168.8 ms       |
| 80, 4000 (improved)        | 6.1 ms (78x)   | 3.1 ms (17x)   | 4.9 ms (34x)   |
| 800, 40000 (original impl) | 19061 ms       | 3951 ms        | 13566 ms       |
| 800, 40000 (improved)      | 30.4 ms (627x) | 19.9 ms (199x) | 20.5 ms (662x) |
| 8000, 400000 (improved)    | 238.5 ms       | 165.5 ms       | 179.5 ms       |

For the small initial `num_ideas` and `num_pkgs` case, the setup is a much higher percentage of process time, so there is diminished improvement multipliers for this section. At higher `num_ideas` and `num_pkgs`, the multipliers are amplified significantly. The `num_ideas=8000` and `num_pkgs=400000` took far too long to execute to measure the improvement.
