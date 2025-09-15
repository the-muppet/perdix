use std::io;
use std::mem;
use std::os::raw::c_int;

use libc::{cpu_set_t, sched_setaffinity, CPU_SET, CPU_ZERO};
use nvml_wrapper::Nvml;

/// Pin current thread to CPUs local to the GPU's NUMA node.
pub fn set_affinity_for_gpu(gpu_index: u32) -> io::Result<()> {
    // init NVML
    let nvml = Nvml::init().map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let device = nvml.device_by_index(gpu_index)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    // query NUMA node (returns Option<u32>)
    let numa_node = device.numa_node()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "NUMA info not available"))?;

    // build CPU set for this NUMA node
    let mut cpuset: cpu_set_t = unsafe { mem::zeroed() };
    unsafe { CPU_ZERO(&mut cpuset) };

    // In practice you’d query `/sys/devices/system/node/node<numa>/cpulist`
    // or use hwloc. For demo, assume logical CPUs 0..N belong.
    for cpu_id in get_cpus_for_numa_node(numa_node)? {
        unsafe { CPU_SET(cpu_id as usize, &mut cpuset) };
    }

    // apply affinity
    let res = unsafe {
        sched_setaffinity(0, mem::size_of::<cpu_set_t>(), &cpuset)
    };
    if res != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn get_cpus_for_numa_node(node: u32) -> io::Result<Vec<u32>> {
    let path = format!("/sys/devices/system/node/node{}/cpulist", node);
    let text = std::fs::read_to_string(path)?;
    parse_cpu_list(&text)
}

/// Parse Linux `cpulist` format (e.g. "0-7,16-23").
fn parse_cpu_list(s: &str) -> io::Result<Vec<u32>> {
    let mut out = Vec::new();
    for part in s.trim().split(',') {
        if let Some((lo, hi)) = part.split_once('-') {
            let lo: u32 = lo.parse().unwrap();
            let hi: u32 = hi.parse().unwrap();
            out.extend(lo..=hi);
        } else {
            out.push(part.parse().unwrap());
        }
    }
    Ok(out)
}

