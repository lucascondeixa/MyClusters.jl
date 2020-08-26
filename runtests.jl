cd(@__DIR__)
push!(LOAD_PATH, dirname(@__DIR__))

include("myclusters_github.jl")
include("test_func.jl")

using DelimitedFiles
using Test

## Test objects
nodes = 3
lines = [[1,3], [2,3]]
Cluss = 1:650
D = Array{Float64}(undef, 8760, nodes)
W = Array{Float64}(undef, 8760, nodes)
S = Array{Float64}(undef, 8760, nodes)
max_load = Array{Float64}(undef, nodes)
dem_inc = Array{Float64}(undef, nodes)
m = zeros(8760)
d = zeros(Int,8760)
pD = 0.01                           # Percentile to find the critical points in Demand
pWS = 0.99                          # Percentile to find the critical points in WS
pace = 0.0005
RP = 50
pD_crit = 0.75
pWS_crit = 0.25
cases = ["SP", "DK", "DE"]

## Exec...
println("\nLoading cases...")
@time load_cases()
println("\nTesting kmeans...")
@time test_kmeans()
println("\nTesting HCnD_WD...")
@time test_HCnD_WD()
println("\nTesting HCnD_ED...")
@time test_HCnD_ED()
println("\nTesting crit_RP...")
@time test_crit_RP()
println("\nTesting find_crit_DWS...")
@time test_find_crit_DWS()

# End of Code
