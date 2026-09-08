"""
Author: Elias Boshara

Julia backend for X-Ray reflectivity simulations using n-box box models and possible inclusion of a protein.
Implementation is optimized for memory and speed while maintaining compatibility with autodifferentation capabilities.
"""

module XPR

using LinearAlgebra
using SpecialFunctions
using ForwardDiff

####### data handling and memory allocation #######

# experimental data and model parameters that won't change during multiple simulations (assuming data fixed)
struct SimData
    data_Q::Vector{Float64}
    data_R::Vector{Float64}
    data_err::Vector{Float64}
    coordinates::Matrix{Float64}
    electrons::Vector{Float64}
    radii::Vector{Float64}
    ρ_top::Float64
    ρ_bottom::Float64
    xray_energy::Float64
    λ::Float64
    r_electron::Float64
    N::Int
    L::Int
    fres::Bool
    max_sd_away::Int
    max_ed_slabs::Int
    max_refl_slabs::Int
    spacez::Float64
    include_protein::Bool
    fit_sig::Bool
    fit_yscl::Bool
    fit_bkg::Bool
    num_boxes::Int
    vol_calc::String
end

# temporary (active) variables (of type T)
mutable struct TempVars{T}
    rot_coordinates::Matrix{T}
    r_radii::Vector{T}
    z_slab::Vector{T}
    α::Vector{T}
    A::T
    events::Vector{T}
    int_pts::Matrix{T}
    slice_coords::Matrix{T}
    slice_radii::Vector{T}
    angles::Vector{T}
    mins::Tuple{T, T}
    pts::Matrix{T}
    cand_locs::Matrix{T}
    cand_rads::Vector{T}
    ρ_protein::Vector{T}
    smear_lattice::Vector{T}
    ρ_hard_z::Vector{T}
    μ::Vector{T}
    w::Vector{T}
    w_mid::Vector{T}
    d::Vector{T}
    ρ::Vector{T}
    β::Vector{T}
    R::Vector{T}
    χ_sq::T
    logL::T
end

# temporary variables specific to area/volume computation of protein
mutable struct AreaVars
    tol::Float64
    cell_max::Int
    max_neighbor::Int
    max_intersect::Int
    intersecting_pairs::Matrix{Int}
    intersect_count::Int
    adjacency::Matrix{Int}
    adjacency_count::Vector{Int}
    intersects::Vector{Bool}
    contained::Vector{Bool}
    counts::Vector{Int}
    pt_owner_circle::Vector{Int}
    circ_to_pt_idx::Matrix{Int}
    pt_is_interior::Vector{Bool}
    pt_count::Int
    grid_xmax::Int
    grid_ymax::Int
    grid_count::Matrix{Int}
    grid::Array{Int, 3}
    cell_size::Float64
end

### global data structs defined for memory reuse across function calls
# global reference for simulation data
const GLOBAL_SD = Ref{Union{Nothing,SimData}}(nothing)
# global reference for area computation variables
const GLOBAL_AV = Ref{Union{Nothing,AreaVars}}(nothing)
# vector of Dicts mapping data types to TempVars
const GLOBAL_TV = Dict{DataType, Any}()

# runs automatically to ensure the dictionary and references are clean on load
function __init__()
    empty!(GLOBAL_TV)
    GLOBAL_AV[] = nothing
    return nothing
end

# for each DataType (float, ForwardDual, etc.), returns or creates unique TempVars{T}
function get_tempvars(T, simdata, areavars_template)
    # Check the single global dictionary directly
    if !haskey(GLOBAL_TV, T) || size(GLOBAL_TV[T].rot_coordinates, 1) < simdata.N
        # allocate for the first time or resize if protein size changed
        tv =  TempVars(
            zeros(T,simdata.N, 3),                                          # rot_coordinates
            zeros(T, simdata.N),                                            # r_radii
            zeros(T, simdata.max_ed_slabs + 1),                             # z_slab                    
            zeros(T, simdata.max_ed_slabs - 2),                             # α
            one(T),                                                         # A
            zeros(T, 2 * simdata.N + 2),                                    # events
            zeros(T, 2*areavars_template.max_intersect, 2),                 # int_pts
            zeros(T, simdata.N, 2),                                         # slice_coords
            zeros(T, simdata.N),                                            # slice_radii
            zeros(T, 2*simdata.N),                                          # angles
            (zero(T), zero(T)),                                             # mins
            zeros(T, simdata.N, 2),                                         # pts
            zeros(T, simdata.N, 2),                                         # cand_locs
            zeros(T, simdata.N),                                            # cand_rads
            zeros(T, simdata.max_ed_slabs),                                 # ρ_protein
            zeros(T, simdata.max_refl_slabs),                               # smear_lattice
            zeros(T, simdata.max_refl_slabs),                               # ρ_hard_z
            zeros(T, simdata.max_refl_slabs),                               # μ: absorption coefficients
            zeros(T, simdata.max_refl_slabs),                               # w
            zeros(T, simdata.max_refl_slabs - 1),                           # w_mid
            zeros(T, simdata.max_refl_slabs - 1),                           # d
            zeros(T, simdata.max_refl_slabs - 1),                           # ρ
            zeros(T, simdata.max_refl_slabs - 1),                           # β
            zeros(T, simdata.L),                                            # R
            zero(T),                                                        # χ_sq
            zero(T)                                                         # logL                                                        
        )
        GLOBAL_TV[T] = tv
    end

    copyto!(GLOBAL_TV[T].r_radii, simdata.radii) # store radii directly 

    return GLOBAL_TV[T]::TempVars{T}
end

# resets global data, should be called whenever protein or data is changed
function reset_data()
    GLOBAL_SD[] = nothing
    GLOBAL_AV[] = nothing
    empty!(GLOBAL_TV)
    return nothing
end

# called once from python at beginning of simulations to import data and load into global data structures
function load_data(data_Q_py, data_R_py, data_err_py, num_boxes, include_protein, fit_sig, fit_yscl, fit_bkg;
    coordinates_py=nothing, electrons_py=nothing, radii_py=nothing, masses_py=nothing, vol_calc = "approx", rho_top=0.0, rho_bottom=0.334)
    
    # reset as needed
    reset_data()
    
    # Convert PyArrays to Julia Arrays
    data_Q = Array(data_Q_py)
    data_R = Array(data_R_py)
    data_err = Array(data_err_py)

    # if protein data input, convert
    if include_protein
        coordinates = Array(coordinates_py)
        electrons = Array(electrons_py)
        radii = Array(radii_py)
        masses = Array(masses_py)
    else
        coordinates = [0.0 0.0 0.0]
        electrons = [1.0]
        radii = [1.0]
        masses = [1.0]
    end
 
    # preallocate 
    simdata, areavars = prealloc(data_Q, data_R, data_err, num_boxes, include_protein, 
    fit_sig, fit_yscl, fit_bkg, coordinates, electrons, radii, masses, vol_calc, rho_top, rho_bottom)
    GLOBAL_SD[] = simdata
    GLOBAL_AV[] = areavars

    return nothing
end

# preallocate memory for simulation data and area computation
function prealloc(data_Q, data_R, data_err, num_boxes, include_protein, 
   fit_sig, fit_yscl, fit_bkg,  coordinates, electrons, radii, masses, vol_calc, rho_top, rho_bottom)

    N = include_protein ? length(radii) : 1                                                  # number of atoms in protein
    L = length(data_Q)                                                                       # length of data set

    spacez = 0.5                                                                             # vertical spacing of grid (Angstroms)

    # given data
    ρ_top        = rho_top                                                                   # electron density of the air
    ρ_bottom     = rho_bottom                                                                # electron density of the buffer
    xray_energy  = 10.0                                                                      # energy of xrays, in inverse units of x
    λ            = 6.62607004e-34 * 2.99792458e8 * 1e10 / xray_energy / 1e3 / 1.60217662e-19 # wavelength of xrays
    r_electron   = 2.818e-5                                                                  # classic radius of an electron

    # include Fresnel normalization
    fres = true
    
    max_sd_away = 10        # how many standard deviations away from interface we extend smearing

    if include_protein
        # calculate total mass and center of mass
        total_mass = sum(masses)
        com = sum(coordinates .* masses, dims=1) ./ total_mass

        # center the coordinate system
        coordinates .-= com

        # compute largest possible grid discretization [max diagonal of bounding box of protein]
        x_min, x_max = extrema(view(coordinates, :, 1))
        y_min, y_max = extrema(view(coordinates, :, 2))
        z_min, z_max = extrema(view(coordinates, :, 3))
        sq_max_l = sqrt((x_max - x_min)^2 + (y_max - y_min)^2 + (z_max - z_min)^2)
        max_r = maximum(radii)
        cell_size = 2.0 * max_r
        max_b = trunc(Int, (sq_max_l / cell_size) + 1)

        # *** decreasing  this runs the risk of seg fault *** #
        # maximum possible phase I slab amount : max protein length + 2 * maximum radius + maximum box lengths + 2 buffers + smearing tails (10*max_sigma / spacez)
        max_ed_slabs = ceil((sq_max_l + 2*max_r + 30*num_boxes) / spacez) + 2 + 2 * ceil(50 / spacez)
    else
        max_ed_slabs = ceil((30*num_boxes) / spacez) + 2 + 2 * ceil(50 / spacez)
        max_b = 1
        cell_size = 1.0
    end
    
    # *** decreasing  this runs the risk of seg fault *** #
    # maximum possible phase II slab amount : max phase I + 2 * (added slabs on one end)
    max_refl_slabs = max_ed_slabs + 2 * ceil((max_sd_away * 5) / spacez)

    # create instance of simulation data
    simdata = SimData(
        data_Q,
        data_R,
        data_err,
        coordinates,
        electrons,
        radii,
        ρ_top,
        ρ_bottom,
        xray_energy,
        λ,
        r_electron,
        N,
        L,
        fres,
        max_sd_away,
        max_ed_slabs,
        max_refl_slabs,
        spacez,
        include_protein,
        fit_sig,
        fit_yscl,
        fit_bkg,
        num_boxes,
        vol_calc
    )

    # area/volume computation values

    # *** decreasing any of the below three values runs the risk of seg fault *** #
    cell_max = simdata.N                                        # number of circles per cell in mesh grid [worst case = N]
    max_intersect = Int(simdata.N * (simdata.N - 1) / 2)        # total number of intersections [worst case = N(N-1)/2]
    max_neighbor = simdata.N                                    # total number of atoms intersecting a given atom [worst case = N]
    
    tol = 1e-8                                                  # threshold to be an interior point
    intersecting_pairs = -ones(Int, max_intersect, 2)           # stores all pairs of circles which intersect
    intersect_count = 0                                         # total interesection count
    adjacency = -ones(Int, simdata.N, max_neighbor)             # tracks which circles are adjacent to each other 
    adjacency_count = zeros(Int, simdata.N)                     # counts number of adjacent circles across data
    intersects = falses(simdata.N)                              # tracks which circles intersect any other
    contained = falses(simdata.N)                               # tracks which circles are fully contained within another
    counts = zeros(Int, simdata.N)                              # tallies number of intersection points on each circle
    pt_owner_circle = zeros(Int, 2*max_intersect)               # indicates which circle "owns" an intersection point
    circ_to_pt_idx = zeros(Int, simdata.N, 2*max_neighbor)      # maps circles to intersection points list
    pt_is_interior = falses(2*max_intersect)                    # tracks if a point is interior
    pt_count = 0                                                # total intersection point counter

    # mesh grid initialization
    grid = zeros(Int, max_b, max_b, cell_max)
    grid_count = zeros(Int, max_b, max_b)
    grid_xmax = 1
    grid_ymax = 1

    # instance of areavars
    areavars = AreaVars(
        tol, 
        cell_max, 
        max_neighbor,
        max_intersect,
        intersecting_pairs, 
        intersect_count,
        adjacency, 
        adjacency_count,
        intersects,
        contained,
        counts,
        pt_owner_circle,
        circ_to_pt_idx,
        pt_is_interior,
        pt_count,
        grid_xmax,
        grid_ymax,
        grid_count,
        grid,
        cell_size
    )

    return simdata, areavars
end

####### helper functions for reflectivity computation #######

# rotate and translate protein
function rotate_protein!(simdata, tempvars, θ, ϕ, d_protein, box_lengths)
    
    # protein already moved to center of mass

    # rows of the rotation matrix
    r1 = [cos(ϕ), -sin(ϕ), 0.0]
    r2 = [cos(θ)*sin(ϕ), cos(θ)*cos(ϕ), -sin(θ)]
    r3 = [sin(θ)*sin(ϕ), sin(θ)*cos(ϕ), cos(θ)]
    rotation_matrix = hcat(r1, r2, r3)'

    # align based on center of mass
    z_offset = d_protein - sum(box_lengths)

    # rotate and translate
    tempvars.rot_coordinates .= (rotation_matrix * simdata.coordinates' .+ [0.0; 0.0; z_offset])'

    ## compute distance of top of protein to bottom box 

    # compute maximum vertical extent of the rotated centered coordinates
    maxh = maximum(vec(r3' * simdata.coordinates') + simdata.radii)
    protein_height = d_protein + maxh

    return protein_height
end

# compute the M+1 slab boundaries, in descending order [assume top slab has thickness zero, final slab is buffer]
function get_slab_boundaries!(simdata, tempvars, box_lengths)

    # top and bottom values of the system
    bot = argmin(tempvars.rot_coordinates[:,3] - simdata.radii)
    bot_val = minimum([tempvars.rot_coordinates[bot,3] - simdata.radii[bot], -sum(box_lengths)])
    top = argmax(tempvars.rot_coordinates[:,3] + simdata.radii)
    top_val = maximum([tempvars.rot_coordinates[top,3] + simdata.radii[top], 0.0])

    # fixed discretization grid
    lo = round(floor(bot_val / simdata.spacez) * simdata.spacez; digits=10)
    hi = round(ceil(top_val / simdata.spacez) * simdata.spacez; digits=10)
    gridz = collect(lo:simdata.spacez:hi)
    M = length(gridz) + 1

    # build grid
    z_slab = zeros(eltype(gridz), length(gridz) + 2)
    z_slab[1] = lo
    z_slab[2:end-1] = gridz
    z_slab[end] = hi

    reverse!(z_slab)

    # store data into tempvars
    for i = eachindex(z_slab)
        tempvars.z_slab[i] = z_slab[i]
    end

    return M
end

# area/volume computation helper 1: find all intersections and adjacency data 
function find_circle_intersections_grid!(areavars, locations, rad)

    offsets = ((0,0), (1,0), (0,1), (1,1), (-1,1)) # need only to check forward neighbors
    xmax, ymax = areavars.grid_xmax, areavars.grid_ymax

    @inbounds for gx in 1:xmax
        @inbounds for gy in 1:ymax
            # current circle
            n_i = areavars.grid_count[gx, gy]
            n_i == 0 && continue

            # iterate through the 5 necessary neighbor cells
            @inbounds for (dx, dy) in offsets
                nx, ny = gx + dx, gy + dy

                # boundary check
                if nx < 1 || nx > xmax || ny < 1 || ny > ymax
                    continue
                end

                n_j = areavars.grid_count[nx, ny]
                n_j == 0 && continue

                same_cell = (dx == 0 && dy == 0)

                @inbounds for gb in 1:n_i
                    i = areavars.grid[gx, gy, gb]
                    if areavars.contained[i] 
                        continue 
                    end
                    
                    xi, yi, ri = locations[i, 1], locations[i, 2], rad[i]

                    # if same cell, start gc at gb + 1 to avoid double counting and i==j
                    start_gc = same_cell ? gb + 1 : 1
                    
                    @inbounds for gc in start_gc:n_j
                        j = areavars.grid[nx, ny, gc]
                        
                        if areavars.contained[j] 
                            continue 
                        end

                        xj, yj, rj = locations[j, 1], locations[j, 2], rad[j]

                        dx_val = xj - xi
                        dy_val = yj - yi
                        dist2 = dx_val * dx_val + dy_val * dy_val
                        
                        sum_r = ri + rj
                        diff_r = ri - rj
                        sum_r2 = sum_r * sum_r
                        diff_r2 = diff_r * diff_r

                        if dist2 >= sum_r2                      # no intersection
                            continue 
                        elseif dist2 <= diff_r2                 # one inside the other
                            if ri < rj
                                areavars.contained[i] = true
                                break
                            else
                                areavars.contained[j] = true
                            end
                        else                                    # proper intersection
                            areavars.intersects[i] = true
                            areavars.intersects[j] = true
                            
                            idx_i = (areavars.adjacency_count[i] += 1)
                            areavars.adjacency[i, idx_i] = j
                            
                            idx_j = (areavars.adjacency_count[j] += 1)
                            areavars.adjacency[j, idx_j] = i
                            
                            p_idx = (areavars.intersect_count += 1)
                            areavars.intersecting_pairs[p_idx, 1] = i
                            areavars.intersecting_pairs[p_idx, 2] = j
                        end
                    end
                end
            end
        end
    end
    return nothing
end

# area/volume computation helper 2: identify and classify intersection points as boundary or interior
function compute_ubp!(tempvars, areavars, locations, rad)
    
    areavars.pt_count = 0

    # iterate through proper intersections, compute and store intersection points
    @inbounds for k in 1:areavars.intersect_count
        i = areavars.intersecting_pairs[k,1]
        j = areavars.intersecting_pairs[k,2]

        if areavars.contained[i] || areavars.contained[j]
            continue
        end

        # data
        x0 = locations[i, 1]
        y0 = locations[i, 2]
        r0 = rad[i]
        x1 = locations[j, 1]
        y1 = locations[j, 2]
        r1 = rad[j]

        # distance vector components
        dx = x1 - x0
        dy = y1 - y0
        d2 = dx * dx + dy * dy
        d  = sqrt(d2)

        # intersection geometry
        a = (r0^2 - r1^2 + d2) / (2 * d)
        h = sqrt(r0^2 - a^2)
        invd = 1.0 / d
        ax = a * dx * invd
        ay = a * dy * invd
        p2x = x0 + ax
        p2y = y0 + ay
        offx = -h * dy * invd
        offy =  h * dx * invd
        
        # store data
        for s in (-1, 1)
            areavars.pt_count += 1
            idx = areavars.pt_count
            px = p2x + s * offx
            py = p2y + s * offy
            
            tempvars.int_pts[idx, 1] = px
            tempvars.int_pts[idx, 2] = py
            
            areavars.counts[i] += 1
            areavars.counts[j] += 1
            areavars.circ_to_pt_idx[i, areavars.counts[i]] = idx
            areavars.circ_to_pt_idx[j, areavars.counts[j]] = idx
            
            areavars.pt_owner_circle[idx] = areavars.adjacency_count[i] <= areavars.adjacency_count[j] ? i : j
        end
    end

    # batch classify as interior / exterior
    @inbounds for p_idx in 1:areavars.pt_count
        c = areavars.pt_owner_circle[p_idx]
        adj_count = areavars.adjacency_count[c]
        px = tempvars.int_pts[p_idx, 1]
        py = tempvars.int_pts[p_idx, 2]
        
        @simd for m = 1:adj_count
            cj = areavars.adjacency[c, m]
            tempvars.cand_locs[m, 1] = locations[cj, 1]
            tempvars.cand_locs[m, 2] = locations[cj, 2]
            tempvars.cand_rads[m]    = rad[cj]
        end

        is_interior = false
        @simd for m = 1:adj_count
            dx = px - tempvars.cand_locs[m, 1]
            dy = py - tempvars.cand_locs[m, 2]
            d2 = dx*dx + dy*dy
            thresh = (tempvars.cand_rads[m] - areavars.tol)^2
            if d2 < thresh
                is_interior = true
            end
        end
        areavars.pt_is_interior[p_idx] = is_interior
    end
    return nothing
end

# area/volume computation helper 3: computing line integral on boundary arcs
function process_arcs!(tempvars, areavars, locations, rad, n)
    
    tol_area = 0.0
    two_pi = 2 * pi

    @inbounds for i = 1:n
        if areavars.contained[i]
            continue
        end

        cand_count = areavars.adjacency_count[i]
        nbpts = areavars.counts[i]
        if cand_count == 0 || nbpts == 0
            continue
        end

        # load candidate data
        @simd for j = 1:cand_count
            cj = areavars.adjacency[i, j]
            tempvars.cand_locs[j, 1] = locations[cj, 1]
            tempvars.cand_locs[j, 2] = locations[cj, 2]
            tempvars.cand_rads[j] = (rad[cj] - areavars.tol)^2 # store squared threshold directly
        end

        # filter for exterior points
        n_out = 0
        for j = 1:nbpts
            pt_idx = areavars.circ_to_pt_idx[i, j]
            if !areavars.pt_is_interior[pt_idx]
                n_out += 1
                tempvars.pts[n_out, 1] = tempvars.int_pts[pt_idx, 1]
                tempvars.pts[n_out, 2] = tempvars.int_pts[pt_idx, 2]
            end
        end

        if n_out == 0
            continue
        end

        # angle computation and sorting
        cx, cy, r = locations[i, 1], locations[i, 2], rad[i]
        @inbounds @simd for j = 1:n_out
            ang = atan(tempvars.pts[j, 2] - cy, tempvars.pts[j, 1] - cx)
            tempvars.angles[j] = ang < 0 ? ang + two_pi : ang
        end
        angles_view = view(tempvars.angles, 1:n_out)
        sort!(angles_view)

        # evaluate the midpoint of every arc to classify as interior or boundary
        for j = 1:n_out
            idx1 = j
            idx2 = (j % n_out) + 1
            
            p1 = angles_view[idx1]
            p2 = angles_view[idx2]
            
            # handle angular wrap-around before calculating midpoint
            if p2 <= p1
                p2 += two_pi
            end
            
            # calculate midpoint of the current arc
            amid = (p1 + p2) / 2.0
            mx, my = cx + r * cos(amid), cy + r * sin(amid)
            
            # check if this specific arc midpoint is inside any candidate circle
            is_interior = false
            for k = 1:cand_count
                dx, dy = tempvars.cand_locs[k, 1] - mx, tempvars.cand_locs[k, 2] - my
                if dx*dx + dy*dy < tempvars.cand_rads[k]
                    is_interior = true
                    break 
                end
            end

            if !is_interior
                # ccw line integral
                r2 = r * r
                term = (r2 / 2.0) * (p2 - p1) + 
                       (r2 / 4.0) * (sin(2 * p2) - sin(2 * p1)) + 
                       (cx * r)   * (sin(p2) - sin(p1))
                tol_area += term
            end
        end
    end
    return tol_area
end

# area/volume computation helper 4: computation of a union of spheres using Green's theorem
function greens!(tempvars, areavars, locations, rad)
    
    n = length(rad)
    if n == 0
        return 0.0
    end
    
    total_area = 0.0

    # reset necessary memory only
    areavars.intersect_count = 0
    fill!(view(areavars.adjacency_count, 1:n), 0)
    fill!(view(areavars.intersects, 1:n), false)
    fill!(view(areavars.contained, 1:n), false)
    fill!(view(areavars.counts, 1:n), 0)
    fill!(areavars.grid_count, 0)
    areavars.pt_count = 0

    # populate grid
    min_x, min_y = tempvars.mins[1], tempvars.mins[2]
    inv_cell_size = 1.0 / areavars.cell_size
    
    @inbounds for idx = 1:n
        gx = floor(Int, (locations[idx, 1] - min_x) * inv_cell_size) + 1
        gy = floor(Int, (locations[idx, 2] - min_y) * inv_cell_size) + 1
        
        # clamp to ensure no bounds errors due to floating point precision
        gx = clamp(gx, 1, areavars.grid_xmax)
        gy = clamp(gy, 1, areavars.grid_ymax)
        
        c = (areavars.grid_count[gx, gy] += 1)
        areavars.grid[gx, gy, c] = idx
    end

    find_circle_intersections_grid!(areavars, locations, rad)

    compute_ubp!(tempvars, areavars, locations, rad)

    # add areas is isolated circles
    @inbounds for i = 1:n
        if !areavars.intersects[i] && !areavars.contained[i]
            total_area += pi * rad[i]^2
        end
    end
    
    total_area += process_arcs!(tempvars, areavars, locations, rad, n)

    return total_area
end

# computes the cross-sectional area for the bottom plane of all slices [depreciated]
function area_slices_nonsmooth!(tempvars, areavars, simdata, M)

    # reset grid specs
    coords = tempvars.rot_coordinates
    x_min, x_max = extrema(view(coords, :, 1))
    y_min, y_max = extrema(view(coords, :, 2))
    tempvars.mins = (x_min, y_min)
    areavars.grid_xmax = trunc(Int, ((x_max - x_min) / areavars.cell_size) + 1)
    areavars.grid_ymax = trunc(Int, ((y_max - y_min) / areavars.cell_size) + 1)

    # total xy-projection area
    total_area = greens!(tempvars, areavars, view(coords, :, 1:2), view(tempvars.r_radii, :))
    tempvars.A = total_area

    # z-slices
    for j in 1:(M - 2)
        # compute circles in the slice
        z_plane = tempvars.z_slab[j + 2]
        n_filtered = 0
        @inbounds for i in 1:simdata.N
            zi = coords[i, 3]
            ri = tempvars.r_radii[i]
            dz = z_plane - zi
            if abs(dz) < ri
                rxy_sq = ri^2 - dz^2
                if rxy_sq > 0
                    n_filtered += 1
                    tempvars.slice_coords[n_filtered, 1] = coords[i, 1]
                    tempvars.slice_coords[n_filtered, 2] = coords[i, 2]
                    tempvars.slice_radii[n_filtered] = sqrt(rxy_sq)
                end
            end
        end
        # compute area for circles in slice
        if n_filtered > 0
            locs_view = view(tempvars.slice_coords, 1:n_filtered, :)
            rads_view = view(tempvars.slice_radii, 1:n_filtered)
            tempvars.α[j] = greens!(tempvars, areavars, locs_view, rads_view)
        else
            tempvars.α[j] = 0.0
        end
    end
    tempvars.α ./= tempvars.A
    return nothing
end

# computes the cross-sectional volume for all slices (event-driven Gaussian quadrature)
function area_slices_gl!(tempvars, areavars, simdata, M)

    T = eltype(tempvars.rot_coordinates)

    # 7-point Gauss-Legendre nodes and weights for interval [-1, 1]
    GL_X = (0.0, 0.4058451513773972, -0.4058451513773972, 0.7415311855993945, -0.7415311855993945, 0.9491079123427585, -0.9491079123427585)
    GL_W = (0.4179591836734694, 0.3818300505051189, 0.3818300505051189, 0.2797053914892766, 0.2797053914892766, 0.1294849661688697, 0.1294849661688697)

    # reset grid specs
    coords = tempvars.rot_coordinates
    x_min, x_max = extrema(view(coords, :, 1))
    y_min, y_max = extrema(view(coords, :, 2))
    tempvars.mins = (x_min, y_min)
    areavars.grid_xmax = trunc(Int, ((x_max - x_min) / areavars.cell_size) + 1)
    areavars.grid_ymax = trunc(Int, ((y_max - y_min) / areavars.cell_size) + 1)

    # total xy-projection area
    total_area = greens!(tempvars, areavars, view(coords, :, 1:2), view(tempvars.r_radii, :))
    tempvars.A = total_area

    # z-slices
    for j in 1:(M - 2)
        z_bot = tempvars.z_slab[j + 2]
        z_top = tempvars.z_slab[j + 1] 
        
        num_events = 0
        
        # always include slab boundaries as integration limits
        num_events += 1 
        tempvars.events[num_events] = z_bot
        num_events += 1 
        tempvars.events[num_events] = z_top

        # identify all topological events within the current slab
        @inbounds for i in 1:simdata.N
            zi = coords[i, 3]
            ri = tempvars.r_radii[i]
            
            # bottom of sphere crosses
            z_event_bot = zi - ri
            if z_bot < z_event_bot < z_top
                num_events += 1
                tempvars.events[num_events] = z_event_bot
            end
            
            # top of sphere crosses
            z_event_top = zi + ri
            if z_bot < z_event_top < z_top
                num_events += 1
                tempvars.events[num_events] = z_event_top
            end
        end

        # sort the events to create sequential integration sub-intervals
        slab_events = view(tempvars.events, 1:num_events)
        sort!(slab_events)

        slab_volume_approx = zero(T)

        # integrate strictly within each smooth sub-interval using Quadrature
        @inbounds for e in 1:(num_events - 1)
            e_start = slab_events[e]
            e_end   = slab_events[e + 1]
            
            dz_sub = e_end - e_start
            
            # skip degenerate intervals (where events occur at virtually the same z)
            if dz_sub > 1e-8
                half_dz = 0.5 * dz_sub
                mid_z   = 0.5 * (e_start + e_end)
                sub_volume = zero(T)
                
                # evaluate the 7-point Gauss-Legendre nodes
                for q in eachindex(GL_X)
                    z_eval = mid_z + half_dz * GL_X[q]
                    n_filtered = 0
                    
                    # compute circles in the current slice
                    for i in 1:simdata.N
                        zi = coords[i, 3]
                        ri = tempvars.r_radii[i]
                        dz_dist = z_eval - zi
                        
                        if abs(dz_dist) < ri
                            rxy_sq = ri^2 - dz_dist^2
                            if rxy_sq > 0
                                n_filtered += 1
                                tempvars.slice_coords[n_filtered, 1] = coords[i, 1]
                                tempvars.slice_coords[n_filtered, 2] = coords[i, 2]
                                tempvars.slice_radii[n_filtered] = sqrt(rxy_sq)
                            end
                        end
                    end
                    # compute area for circles in slice and add to quadrature sum
                    if n_filtered > 0
                        locs_view = view(tempvars.slice_coords, 1:n_filtered, :)
                        rads_view = view(tempvars.slice_radii, 1:n_filtered)
                        area_q = greens!(tempvars, areavars, locs_view, rads_view)
                        
                        sub_volume += GL_W[q] * area_q
                    end
                end
                # scale by the Jacobian of the interval transformation
                slab_volume_approx += half_dz * sub_volume
            end
        end
        tempvars.α[j] = slab_volume_approx
    end
    tempvars.α ./= (tempvars.A * simdata.spacez)
    return nothing
end

# computes the cross-sectional volume for all slices (method of spherical segments; double-counts, coverage param range should be [0,10])
function area_slices_sphere_seg!(tempvars, simdata, M)

    T = eltype(tempvars.rot_coordinates)

    # cross-sectional area
    total_area = zero(T)
    @inbounds @simd for i in 1:simdata.N
        total_area += T(π) * tempvars.r_radii[i]^2
    end
    tempvars.A = total_area

    @inbounds for j = 3:M # iterate through non-boundary slabs
        top_bound = tempvars.z_slab[j-1]
        bot_bound = tempvars.z_slab[j]
        
        d_acc = 0.0 

        @simd for i = 1:simdata.N # iterate through atoms
            zi = tempvars.rot_coordinates[i,3]
            ri = tempvars.r_radii[i]
            
            zrp = zi + ri
            zrm = zi - ri
            
            z_ij_top = ifelse(zrp < top_bound, zrp, top_bound)
            z_ij_bot = ifelse(zrm > bot_bound, zrm, bot_bound)

            in_bounds = (top_bound > zrm) && (bot_bound < zrp)
            h_ij = ifelse(in_bounds, z_ij_top - z_ij_bot, 0.0)

            d = z_ij_bot - zi
            num = h_ij * (ri^2 - d^2 - h_ij*d - h_ij^2 / 3)
            denom = (4 / 3) * ri^3
            
            d_acc += num / denom
        end
        tempvars.α[j-2] = (d_acc) / (tempvars.A * (top_bound - bot_bound))
    end
    return nothing
end

# computes the electron densities at each slab using the method of spherical segments
function electron_densities!(simdata, tempvars, M)

    @inbounds for j = 3:M # iterate through non-boundary slabs
        top_bound = tempvars.z_slab[j-1]
        bot_bound = tempvars.z_slab[j]
        
        ρ_acc = 0.0 

        @simd for i = 1:simdata.N # iterate through atoms
            zi = tempvars.rot_coordinates[i,3]
            ri = simdata.radii[i]
            
            zrp = zi + ri
            zrm = zi - ri
            
            z_ij_top = ifelse(zrp < top_bound, zrp, top_bound)
            z_ij_bot = ifelse(zrm > bot_bound, zrm, bot_bound)

            in_bounds = (top_bound > zrm) && (bot_bound < zrp)
            h_ij = ifelse(in_bounds, z_ij_top - z_ij_bot, 0.0)

            d = z_ij_bot - zi
            num = h_ij * (ri^2 - d^2 - h_ij*d - h_ij^2 / 3)
            denom = (4 / 3) * ri^3
            
            ρ_acc += simdata.electrons[i] * num / denom
        end
        tempvars.ρ_protein[j-1] = ρ_acc / (tempvars.A * (top_bound - bot_bound))
    end
    return nothing
end

# arbitrary step function, size(x_int) = size(yvals) + 1, x_int should be in decreasing order
function step_function(x, x_int, yvals)

    nv = length(yvals)

    T = promote_type(eltype(x), eltype(x_int), eltype(yvals))
    result = zeros(T, length(x))

    @inbounds for (i, xi) in enumerate(x)
        val = zero(T)
        if xi <= x_int[1] && xi >= x_int[end]
            val = yvals[end]
            @inbounds for ii in 1:nv
                if xi <= x_int[ii] && xi > x_int[ii+1]
                    val = yvals[ii]
                    break
                end
            end
        end
        result[i] = val
    end

    return result
end

# computes the relevant values of ρ_hard (i.e. all unique values), intervals, and combined interface
function smear_xy_interface!(simdata, tempvars, box_lengths, box_densities, C, M, sig)
    
    sig_val = sig isa ForwardDiff.Dual ? ForwardDiff.value(sig) : sig
    
    if simdata.include_protein
        tv = maximum([0.0, tempvars.z_slab[1]])
        bv = minimum([-sum(box_lengths), tempvars.z_slab[M + 1]])
    else
        tv = 0.0
        bv = -sum(box_lengths)
    end
    
    tv_val = tv isa ForwardDiff.Dual ? ForwardDiff.value(tv) : tv
    bv_val = bv isa ForwardDiff.Dual ? ForwardDiff.value(bv) : bv
    
    # generate unified rigid lattice spanning the entire padded simulation
    lo = round(floor((bv_val - simdata.max_sd_away * sig_val) / simdata.spacez) * simdata.spacez; digits=10)
    hi = round(ceil((tv_val + simdata.max_sd_away * sig_val) / simdata.spacez) * simdata.spacez; digits=10)
    gridz = collect(lo:simdata.spacez:hi)
    
    # inject interfaces into the unified grid, remove redundant overlapping points
    rall = sort(unique([gridz; 0.0; -cumsum(box_lengths)]))
    
    for i = eachindex(rall)
        tempvars.smear_lattice[i] = rall[i]
    end

    # step function eval points
    B = length(rall)
    smear_mpt = similar(tempvars.smear_lattice, B - 1)
    @simd for i = 1:(B-1)
        smear_mpt[i] = (tempvars.smear_lattice[i] + tempvars.smear_lattice[i+1]) * 0.5
    end

    # build the discretization blocks for smearing and reflectivity in ascending order
    for i = eachindex(rall)
        tempvars.w[i] = rall[i]
    end
    for i = 1:(B-1)
        tempvars.w_mid[i] = (tempvars.w[i] + tempvars.w[i+1]) * 0.5
        tempvars.d[i] = tempvars.w[i+1] - tempvars.w[i] 
    end

    # step function computation
    ρ_box_z = step_function(smear_mpt, [Inf; 0.0; -cumsum(box_lengths); -Inf], [simdata.ρ_top; box_densities; simdata.ρ_bottom])
    if simdata.include_protein
        ρ_protein_z = step_function(smear_mpt, view(tempvars.z_slab, 1:(M+1)), view(tempvars.ρ_protein, 1:M))
        α_z = step_function(smear_mpt, view(tempvars.z_slab, 2:M), view(tempvars.α, 1:(M-2)))

        # combined interface function for protein
        temp_ρhz = C .* ρ_protein_z .+ (1 .- C .* α_z) .* ρ_box_z
    else
        temp_ρhz = ρ_box_z
    end
    for i in eachindex(temp_ρhz)
        tempvars.ρ_hard_z[i] = temp_ρhz[i]
    end
    return B
end

# evaluated the Gaussian-smeared profile at z (scalar or vector) using hard-interface x and y values, and smearing factor
function eval_smeared!(tempvars, B, sig)

    μ = view(tempvars.μ, 1:(B-1))

    T = eltype(tempvars.smear_lattice)
    sqrt2sig = sqrt(T(2))*sig

    # do this for both ρ [electron density] and β [absorption density]
    @inbounds for i = 1:(B-1)
        zi = tempvars.w_mid[i]
        ρ1 = zero(T)
        ρ2 = zero(T)
        @simd for j = 2:(B-1)
            ρ1 += (tempvars.ρ_hard_z[j] - tempvars.ρ_hard_z[j-1]) * erf((zi - tempvars.smear_lattice[j]) / sqrt2sig)
            ρ2 += (μ[j] - μ[j-1]) * erf((zi - tempvars.smear_lattice[j]) / sqrt2sig)
        end
        ρ1 += (tempvars.ρ_hard_z[1] + tempvars.ρ_hard_z[(B-1)])
        ρ2 += (μ[1] + μ[(B-1)])
        tempvars.ρ[i] = max(ρ1 / 2, zero(T))
        tempvars.β[i] = max(ρ2 / 2, zero(T))
    end

    return nothing
end

# reflectivity with Fresnel normalization and shift/scaling
function reflectivity!(simdata, tempvars, q_offset, B, yscl, bkg)

    # compute Parratt reflectivity of interface using descending views
    tempR = parratt_reflectivity(simdata, view(tempvars.ρ, (B-1):-1:1), view(tempvars.β, (B-1):-1:1), view(tempvars.d, (B-1):-1:1), q_offset)
    for i in eachindex(tempvars.R)
        tempvars.R[i] = tempR[i]
    end

    if simdata.fres
        # Parratt reflectivity of empty experiment
        R_fr = parratt_reflectivity(simdata, [simdata.ρ_top; simdata.ρ_bottom], [0; 0], [0; 1], 0)
        # Fresnel normalization
        for i in eachindex(tempvars.R)
            tempvars.R[i] = tempvars.R[i] / R_fr[i]
        end
    end

    tempvars.R .= yscl * tempvars.R .+ bkg # scale

    return nothing
end

# iterative implementation of the Parratt reflectivity
function parratt_reflectivity(simdata, ρ_f, β_f, d_f, q_offset_f)
    B_f = length(ρ_f)
    Rc = zeros(eltype(ρ_f), simdata.L) * 1im    # necessary for autodiff
    R_f = zeros(eltype(ρ_f), simdata.L)         # necessary for autodiff
    c1 = 16 * π * simdata.r_electron
    c2 = -32 * π^2 / simdata.λ^2
    @inbounds for l = 1:simdata.L
        Ql = simdata.data_Q[l] + q_offset_f
        @inbounds for it = 0:(B_f-2)
            k = B_f - it
            Qc_km1 = c1 * (ρ_f[k-1] - ρ_f[1])
            Qc_k = c1 * (ρ_f[k] - ρ_f[1])
            S_km1 = sqrt(Ql^2 - Qc_km1 + β_f[k-1]*c2*1im)
            S_k = sqrt(Ql^2 - Qc_k + β_f[k]*c2*1im)
            r_k = (S_km1 - S_k) / (S_km1 + S_k)
            eSd = exp(S_k * d_f[k] * 1im)
            Rc[l] = (r_k + Rc[l] * eSd) / (1.0 + r_k * Rc[l] * eSd)
        end
        R_f[l] = abs2(Rc[l]) 
    end
    return R_f
end

# main function call, returns simluated data, and log-likelihood of simulated data
function XPR!(params)

    # get memory structs
    T = eltype(params)
    simdata = GLOBAL_SD[]
    areavars = GLOBAL_AV[] 
    tempvars = get_tempvars(T, simdata, areavars)

    # fitted parameters
    q_offset  = params[1]
    θ         = params[2] * π / 180 #convert to radians
    ϕ         = params[3] * π / 180 #convert to radians
    d_protein = params[4]
    C         = params[5] 
    sig       = simdata.fit_sig ? params[6] : 3.4 # default roughness parameter
    yscl      = simdata.fit_yscl ? params[7] : 1.0 # default multiplicative scaling parameter of simulated reflectivity
    bkg       = simdata.fit_bkg ? params[8] : 0.0  # default additive scaling parameter of simulated reflectivity

    nbox = simdata.num_boxes
    # boxes order from top to bottom (box one sits nearest to top slab, box n nearest to buffer/bulk slab)
    box_lengths = params[9:nbox+8]
    box_densities = params[nbox+9: 2*nbox+8]
    
    if simdata.include_protein
        # rotate the protein and align z-height
        protein_height = rotate_protein!(simdata, tempvars, θ, ϕ, d_protein, box_lengths)

        # get slab boundaries [add redundant slabs on either side of protein-lipid], in descending order
        M = get_slab_boundaries!(simdata, tempvars, box_lengths)

        # compute projected area of protein, area slices
        if simdata.vol_calc == "exact"
            area_slices_gl!(tempvars, areavars, simdata, M)
        elseif simdata.vol_calc == "approx"
            area_slices_sphere_seg!(tempvars, simdata, M)
        end

        # compute electron densities of protein
        electron_densities!(simdata, tempvars, M)
    else
        M = 1
        protein_height = 0.0
    end
    
    # obtain the values of ρ_hard, along with boundaries, needed for smearing
    B = smear_xy_interface!(simdata, tempvars, box_lengths, box_densities, C, M, sig)

    # evaluate the smeared ρ and μs
    eval_smeared!(tempvars, B, sig)

    # reflectivity computaiton
    reflectivity!(simdata, tempvars, q_offset, B, yscl, bkg)

    # goodness of fit values
    tempvars.χ_sq = sum(((simdata.data_R - tempvars.R) ./ simdata.data_err).^2)
    tempvars.logL = -0.5 * tempvars.χ_sq

    return tempvars.logL, tempvars.R, protein_height
end

# wrapper function to help gradient computation, return logL only
function XPR_grad_helper(params)
    result = XPR!(params)
    return result[1]
end

####### main function and derivative callers #######

# wrapper, returns all values of XPR
function XPR_sim_ref(params)
    params = Array(params) # convert PyArray to Julia Array
    logL, R, prot_h = XPR!(params)
    return logL, R, prot_h
end

# gradient computed via forward differentiation
function XPR_grad(params)
    params = Array(params) # convert PyArray to Julia Array
    grad = ForwardDiff.gradient(p -> XPR_grad_helper(p), params)
    return grad
end

# Hessian computed via forward differentiation
function XPR_hess(params)
    params = Array(params) # convert PyArray to Julia Array
    hess = ForwardDiff.hessian(p -> XPR_grad_helper(p), params)
    return hess
end

####### exported functions #######

export load_data
export XPR_sim_ref
export XPR_grad
export XPR_hess

end