mod commands;
mod dto;
mod face_cache;
mod model_manager;
mod state;

use state::AppState;
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::new())
        .setup(|app| {
            #[cfg(target_os = "macos")]
            {
                // Native macOS menu enables Cmd+C/V/X/A/Z in WKWebView inputs and standard Cmd+Q/M
                if let Ok(menu) = tauri::menu::Menu::default(app.handle()) {
                    let _ = app.set_menu(menu);
                }
            }
            if let Ok(app_data_dir) = app.path().app_data_dir() {
                let state = app.state::<AppState>();
                state.set_managed_models_dir(app_data_dir.join("models"));
            }
            // Failsafe: ensure main window reveals even if WebView startup is unusually delayed
            let app_handle = app.handle().clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(1200));
                if let Some(window) = app_handle.get_webview_window("main")
                    && !window.is_visible().unwrap_or(true)
                {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::exit_app,
            commands::reveal_main_window,
            commands::check_runtime_status,
            commands::check_models_manifest,
            commands::download_missing_models,
            commands::cancel_model_downloads,
            commands::set_gpu_acceleration,
            commands::get_storage_cache_summary,
            commands::set_storage_cache_dir,
            commands::clear_storage_cache,
            commands::inspect_onnx_model,
            commands::set_custom_ai_models,
            commands::load_shoot_folder,
            commands::run_face_clustering,
            commands::scan_project_slates_ocr,
            commands::cancel_slate_ocr_scan,
            commands::scan_single_photo_ocr,
            commands::apply_slate_ocr_renames,
            commands::rename_cluster,
            commands::merge_clusters,
            commands::split_photos_to_new_cluster,
            commands::move_photos_to_cluster,
            commands::create_virtual_folder,
            commands::rename_virtual_folder,
            commands::move_virtual_folder,
            commands::delete_virtual_folder,
            commands::assign_photos_to_virtual_folder,
            commands::resolve_ambiguous_match,
            commands::undo_organization_action,
            commands::redo_organization_action,
            commands::auto_level_cluster_eyes,
            commands::load_develop_working_buffer,
            commands::generate_ai_mask,
            commands::save_photo_develop_recipe,
            commands::save_photo_develop_recipes_batch,
            commands::sync_cluster_develop_recipe,
            commands::apply_relative_develop_nudge,
            commands::calibrate_wb_from_slate,
            commands::estimate_export_size,
            commands::organize_and_export_shoot,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Kroma desktop application");
}
