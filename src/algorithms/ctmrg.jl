@with_kw struct CTMRG #<: Algorithm
    trscheme::TruncationScheme = TensorKit.notrunc()
    tol::Float64 = Defaults.tol
    maxiter::Integer = Defaults.maxiter
    verbose::Integer = 0
end

function MPSKit.leading_boundary(peps::InfinitePEPS,alg::CTMRG,envs = CTMRGEnv(peps))
    err = Inf
    iter = 0

    #for convergence criterium we use the on site contracted boundary
    #this convergences, though the value depends on the bond dimension χ
    old_norm = abs(contract_ctrmg(peps,envs,1,1))

    while err > alg.tol && iter <= alg.maxiter

        for dir in 1:4
            envs = left_move(peps,alg,envs);

            envs = rotate_north(envs,EAST);
            peps = envs.peps;
        end
        new_norm = abs(contract_ctrmg(peps,envs,1,1))

        err = abs(old_norm-new_norm)
        @ignore_derivatives alg.verbose > 0 && @info "iter $(iter): error = $(err)"
        

        old_norm = new_norm
        iter += 1
    end
    @ignore_derivatives iter > alg.maxiter && @warn "maxiter $(alg.maxiter) reached: error was $(err)"

    envs
end

# the actual left_move is dependent on the type of ctmrg, so this seems natural
function left_move(peps::InfinitePEPS{PType},alg::CTMRG,envs::CTMRGEnv) where PType
    corners::typeof(envs.corners) = copy(envs.corners);
    edges::typeof(envs.edges) = copy(envs.edges);

    above_projector_type = tensormaptype(spacetype(PType),1,3,storagetype(PType));
    below_projector_type = tensormaptype(spacetype(PType),3,1,storagetype(PType));

    for col in 1:size(peps,2)

        above_projs = Vector{above_projector_type}(undef,size(peps,1));
        below_projs = Vector{below_projector_type}(undef,size(peps,1));

        # find all projectors
        for row in 1:size(peps,1)
            peps_nw = peps[row,col];
            peps_sw = rotate_north(peps[row+1,col],WEST);


            Q1 = northwest_corner(envs.edges[SOUTH,row+1,col],envs.corners[SOUTHWEST,row+1,col],envs.edges[WEST,row+1,col],peps_sw);
            Q2 = northwest_corner(envs.edges[WEST,row,col],envs.corners[NORTHWEST,row,col],envs.edges[NORTH,row,col],peps_nw);
            (U,S,V) = tsvd(Q1*Q2,alg=SVD(),trunc = alg.trscheme);

            isqS = sdiag_inv_sqrt(S);
            Q = isqS*U'*Q1;
            P = Q2*V'*isqS;

            @diffset above_projs[row] = Q;
            @diffset below_projs[row] = P;
        end

        #use the projectors to grow the corners/edges
        for row in 1:size(peps,1)
            Q = above_projs[row];
            P = below_projs[mod1(row+1,end)];

        
            @diffset @tensor corners[NORTHWEST,row+1,col+1][-1;-2] := envs.corners[NORTHWEST,row+1,col][1,2] * envs.edges[NORTH,row+1,col][2,3,4,-2]*Q[-1;1 3 4]
            @diffset @tensor corners[SOUTHWEST,row+1,col+1][-1;-2] := envs.corners[SOUTHWEST,row+1,col][1,4] * envs.edges[SOUTH,row+1,col][-1,2,3,1]*P[4 2 3;-2]
            @diffset @tensor edges[WEST,row,col+1][-1 -2 -3;-4] := envs.edges[WEST,row,col][1 2 3;4]*peps[row,col][9;5 -2 7 2]*conj(peps[row,col][9;6 -3 8 3])*P[4 5 6;-4]*Q[-1;1 7 8]

        end

        @diffset corners[NORTHWEST,:,col+1]./=norm.(corners[NORTHWEST,:,col+1]);
        @diffset corners[SOUTHWEST,:,col+1]./=norm.(corners[SOUTHWEST,:,col+1]);
        @diffset edges[WEST,:,col+1]./=norm.(edges[WEST,:,col+1]);
    end
    
    CTMRGEnv(peps,corners,edges);
end

function contract_ctrmg(peps::InfinitePEPS{PType},envs::CTMRGEnv,i::Integer,j::Integer) where PType
    peps_nw = peps[i,j];
    Q2 = northwest_corner(envs.edges[WEST,i,j],envs.corners[NORTHWEST,i,j],envs.edges[NORTH,i,j],peps_nw);
    contracted = @tensor Q2[1 2 3;6 4 5]*envs.corners[SOUTHWEST,i,j][7;1]*envs.edges[SOUTH,i,j][8,2,3;7]*envs.corners[SOUTHEAST,i,j][9;8]*envs.edges[EAST,i,j][10,4,5;9]*envs.corners[NORTHEAST,i,j][6;10]
end

northwest_corner(E4,C1,E1,peps) = @tensor corner[-1 -2 -3;-4 -5 -6] := E4[-1 1 2;3]*C1[3;4]*E1[4 5 6;-4]*peps[7;5 -5 -2 1]*conj(peps[7;6 -6 -3 2])
