push!(LOAD_PATH, dirname(@__DIR__))

using Documenter
using MyClusters

makedocs(
    sitename = "MyClusters",
    format = Documenter.HTML(),
    modules = [MyClusters]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#

deploydocs(
    repo = "github.com/lucascondeixa/MyClusters.jl.git",
)
