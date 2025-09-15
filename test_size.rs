#[repr(C)]
struct Slot {
    seq: u64,
    len: u32,
    flags: u32,
    _pad1: u32,
    payload: [u8; 240],
}

fn main() {
    println!("Slot size: {} bytes", std::mem::size_of::<Slot>());
    println!("Slot alignment: {} bytes", std::mem::align_of::<Slot>());
}
