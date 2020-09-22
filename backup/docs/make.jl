# NOTE: If your source directory is not accessible through Julia's LOAD_PATH,
# you might wish to add the following line at the top of make.jl

push!(LOAD_PATH, dirname(@__DIR__))

using Documenter, MyClusters

makedocs(
    sitename = "MyClusters",
    format = Documenter.HTML(),
    modules = [MyClusters]
)
