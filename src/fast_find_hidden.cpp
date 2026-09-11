#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <chrono>

namespace fs = std::filesystem;

struct HiddenEntry {
    std::string type;
    std::string path;
};

void scan_directory(const fs::path& root, const fs::path& current, std::vector<HiddenEntry>& entries) {
    try {
        if (!fs::exists(current) || !fs::is_directory(current)) return;

        for (const auto& entry : fs::directory_iterator(current, fs::directory_options::skip_permission_denied)) {
            std::string filename = entry.path().filename().string();
            fs::path rel_path = fs::relative(entry.path(), root);
            std::string rel_str = rel_path.string();

            // Skip internal git blob objects directory tree for speed
            if (rel_str.find(".git/objects") != std::string::npos || rel_str.find(".git/hooks") != std::string::npos) {
                continue;
            }

            bool is_hidden = (!filename.empty() && filename[0] == '.');

            if (is_hidden) {
                if (entry.is_directory()) {
                    entries.push_back({"DIRECTORY", rel_str});
                } else if (entry.is_regular_file() || entry.is_symlink()) {
                    entries.push_back({"FILE", rel_str});
                }
            }

            if (entry.is_directory() && !entry.is_symlink()) {
                scan_directory(root, entry.path(), entries);
            }
        }
    } catch (const std::exception& e) {
        // Handle permission or filesystem errors gracefully
    }
}

int main(int argc, char* argv[]) {
    std::string root_str = (argc > 1) ? argv[1] : ".";
    fs::path root = fs::canonical(root_str);

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<HiddenEntry> hidden_entries;
    scan_directory(root, root, hidden_entries);

    std::sort(hidden_entries.begin(), hidden_entries.end(), [](const HiddenEntry& a, const HiddenEntry& b) {
        return a.path < b.path;
    });

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "=== Native C++ Fast Hidden File Scanner (Zero os.walk) ===" << std::endl;
    std::cout << "Root: " << root.string() << std::endl;
    std::cout << "Scan Time: " << elapsed_ms << " ms" << std::endl;
    std::cout << "Total Hidden Items Found: " << hidden_entries.size() << std::endl << std::endl;

    for (const auto& entry : hidden_entries) {
        std::cout << "[" << entry.type << "] " << entry.path << "\n";
    }

    return 0;
}
