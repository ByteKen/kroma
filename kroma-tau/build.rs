fn main() {
    tauri_build::build();

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        // CoreML EP in onnxruntime requires macOS 13.4+ deployment target
        println!("cargo:rustc-link-arg=-mmacosx-version-min=13.4");

        // Clang runtime helper for ___isPlatformVersionAtLeast (__builtin_available)
        let mut found_rt = false;
        if let Ok(output) = std::process::Command::new("clang")
            .arg("-print-libgcc-file-name")
            .output()
            && output.status.success()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let rt_path = std::path::Path::new(stdout.trim());
            if rt_path.exists() {
                println!("cargo:rustc-link-arg={}", rt_path.display());
                if let Some(parent) = rt_path.parent() {
                    println!("cargo:rustc-link-search=native={}", parent.display());
                    println!("cargo:rustc-link-lib=clang_rt.osx");
                }
                found_rt = true;
            }
        }

        if !found_rt {
            // Fallback search across common macOS Command Line Tools / Xcode paths
            let candidates = [
                "/Library/Developer/CommandLineTools/usr/lib/clang",
                "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang",
            ];
            for base in &candidates {
                if let Ok(entries) = std::fs::read_dir(base) {
                    for entry in entries.flatten() {
                        let candidate_file = entry.path().join("lib/darwin/libclang_rt.osx.a");
                        if candidate_file.exists() {
                            println!("cargo:rustc-link-arg={}", candidate_file.display());
                            if let Some(parent) = candidate_file.parent() {
                                println!("cargo:rustc-link-search=native={}", parent.display());
                                println!("cargo:rustc-link-lib=clang_rt.osx");
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
}

