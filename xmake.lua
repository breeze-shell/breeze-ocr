
set_project("breeze-ocr")
set_policy("compatibility.version", "3.0")

includes("deps/ncnn.lua")

add_requires("ncnn", "yalantinglibs", {
    configs = {
        ssl = true
    }
})

set_languages("c++2b")
set_warnings("all")
add_rules("plugin.compile_commands.autoupdate", {outputdir = "build"})
add_rules("mode.releasedbg")

if is_plat("windows") then
    add_defines("NOMINMAX")
end

target("breeze-ocr")
    set_kind("binary")
    set_encodings("utf-8")
    add_packages("ncnn", "yalantinglibs")
    add_files("src/*.cc", "src/*/**.cc") 
