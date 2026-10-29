function [ n_nu, i_nu, r, mN, gN, r_allall, mat ] = set_crystal_v2( crystal_mode, mu_n, r_en_limit, r_input, mat_input, structdata)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

aaa = 0; %not used
switch crystal_mode
    case 1 %指定
        [r, mat] = load_crystal(structdata.symmetry, r_en_limit, structdata.latticeconst, structdata.nucpara, r_input, mat_input, structdata.hdirection);
        n_nu = r.number;									  %%number of nuclear spin site
        r_allall = r;
        r = r.all;		                    %% position of nuclear spin, electron spin is set at r=0
        i_nu = zeros(n_nu,1);
        mN = ones(n_nu,1)*mu_n;
        gN = zeros(n_nu,3);
        for k=1:n_nu
            i_nu(k,1) = ones(1,1)*nucspin(mat.all_spec{k});
            gN(k,:) = ones(1,3)*nucgval(mat.all_spec{k}); 					    %% (diagonal part of) gfactor tensor of nuclear spin
        end
        
    case 2 %ファイルを読み取る場合にはlattice const動かないことに注意。
        [r, mat] = load_crystal(structdata.filename, r_en_limit, aaa, structdata.nucpara, r_input, mat_input, structdata.hdirection);
        n_nu = r.number;									  %%number of nuclear spin site
        r_allall = r;
        r = r.all;		                    %% position of nuclear spin, electron spin is set at r=0
        i_nu = zeros(n_nu,1);
        mN = ones(n_nu,1)*mu_n;
        gN = zeros(n_nu,3);
        for k=1:n_nu
            i_nu(k,1) = ones(1,1)*nucspin(mat.all_spec{k});
            gN(k,:) = ones(1,3)*nucgval(mat.all_spec{k}); 					    %% (diagonal part of) gfactor tensor of nuclear spin
        end
        
    case 3 %ファイルを読み取る場合にはlattice const動かないことに注意。
        z_direction = structdata.hdirection;
        
%                 [r, mat] = load_crystal('mp_data/CeO2_mp-20194_primitive.cif',r_en_limit,aaa,{{'O' [17]} {'Ce' [136] [0]}}, r_input, mat_input, z_direction);
%                 [r, mat] = load_crystal('mp_data/CaO_mp-2605_primitive.cif',r_en_limit,aaa,{{'Ca' [43]} {'O' [17]}}, r_input, mat_input, z_direction);
%                 [r, mat] = load_crystal('mp_data/SiO2_mp-6930_primitive.cif',r_en_limit,aaa,{{'O' [17]} {'Si' [29]}}, r_input, mat_input, z_direction);
%                 [r, mat] = load_crystal('mp_data/Y2SiO5_mp-3520_primitive.cif',r_en_limit,aaa,{{'Y' [89]} {'Si' [29]} {'O' [17]}}, r_input, mat_input, z_direction);
%                 [r, mat] = load_crystal('mp_data/ZnO_mp-2133_primitive.cif',r_en_limit,aaa,{{'O' [17]} {'Zn' [67]}}, r_input, mat_input, z_direction);
%         % 3C SiC
%                         [r, mat] = load_crystal('mp_data/SiC_mp-8062_primitive.cif',r_en_limit,0.5,{{'C' [13]} {'Si' [29]}}, r_input, mat_input, z_direction);
%         % 4H SiC
        %         [r, mat] = load_crystal('mp_data/SiC_mp-11714_primitive.cif',r_en_limit,0.5,{{'C' [13]} {'Si' [29]}}, r_input, mat_input, z_direction);
%         % 6H SiC
%                 [r, mat] = load_crystal('mp_data/SiC_mp-7631_primitive.cif',r_en_limit,0.5,{{'C' [13]} {'Si' [29]}}, r_input, mat_input, z_direction);
%         [r, mat] = load_crystal('mp_data/C_mp-66_primitive.cif',r_en_limit,aaa, {{'C' [13]}}, r_input, mat_input, structdata.hdirection);
%                 [r, mat] = load_crystal('mp_data/MgO_mp-1265_primitive.cif',r_en_limit,aaa,{{'Mg' [25]} {'O' [17]}}, r_input, mat_input, z_direction);
        %         [r, mat] = load_crystal('mp_data/BeO_mp-2542_primitive.cif',r_en_limit,aaa,{{'Be' [9] [0.01]} {'O' [17]}}, r_input, mat_input, z_direction);
        %         [r, mat] = load_crystal('mp_data/Al2O3_mp-1143_primitive.cif',r_en_limit,aaa,{{'Al' [27]} {'O' [17]}}, r_input, mat_input, z_direction);
        %         [r, mat] = load_crystal('mp_data/Mg2SiO4_mp-2895_primitive.cif',r_en_limit,aaa,{{'Mg' [25]} {'Si' [29]} {'O' [17]}}, r_input, mat_input, z_direction);
        
        %         [r, mat] = load_crystal('mp_data/ZnO_mp-2229_primitive.cif',r_en_limit,aaa,{{'O' [17]} {'Zn' [67]}}, r_input, mat_input, z_direction);
        %         [r, mat] = load_crystal('mp_data/ZnS_mp-10695_primitive.cif',r_en_limit,aaa,{{'S' [33]} {'Zn' [67]}}, r_input, mat_input, z_direction);
        %         [r, mat] = load_crystal('mp_data/AlN_mp-661_primitive.cif',r_en_limit,aaa,{{'Al' [27]} {'N' [14]} {'N' [15]}}, r_input, mat_input, z_direction);
        %         [r, mat] = load_crystal('mp_data/TiO2_mp-2657_primitive.cif',r_en_limit,aaa,{{'O' [17]} {'Ti' [47]} {'Ti' [49]}}, r_input, mat_input, z_direction);
                [r, mat] = load_crystal('mp_data/Si_mp-149_primitive.cif',r_en_limit,aaa,{{'Si' [29]}}, r_input, mat_input, z_direction);
        %         [r, mat] = load_crystal('mp_data/ZnO_mp-2133_primitive.cif',r_en_limit,aaa,{{'O' [17]} {'Zn' [67]}}, r_input, mat_input, z_direction);
%                 [r, mat] = load_crystal('mp_data/SiO2_mp-7000_primitive.cif',r_en_limit,aaa,{{'O' [17]} {'Si' [29]}}, r_input, mat_input, z_direction);
        %         [r, mat] = load_crystal('mp_data/C_mp-568286_primitive.cif',r_en_limit,aaa,{{'C' [13]} }, r_input, mat_input, z_direction);
        %         [r, mat] = load_crystal('mp_data/ZnO_mp-2133_primitive.cif',r_en_limit,aaa,{{'O' [17]} {'Zn' [67]}}, r_input, mat_input, z_direction);
        %         [r, mat] = load_crystal('mp_data/SiO2_mp-546794_primitive.cif',r_en_limit,aaa,{{'O' [17]} {'Si' [29]}}, r_input, mat_input, z_direction);
        %         [r, mat] = load_crystal('mp_data/SiO2_mp-546794_primitive.cif',r_en_limit,aaa,{{'O' [17]} {'Si' [29]}}, r_input, mat_input, z_direction);
        %         [r, mat] = load_crystal('mp_data/SiO2_mp-546794_primitive.cif',r_en_limit,aaa,{{'O' [17]} {'Si' [29]}}, r_input, mat_input, z_direction);
        %         [r, mat] = load_crystal('mp_data/SiO2_mp-546794_primitive.cif',r_en_limit,aaa,{{'O' [17]} {'Si' [29]}}, r_input, mat_input, z_direction);
        %         [r, mat] = load_crystal('mp_data/GaAs_mp-2534_primitive.cif',r_en_limit,aaa,{{'Ga' [69]} {'Ga' [71]} {'As' [75]}}, r_input, mat_input, z_direction);
        %         [r, mat] = load_crystal('mp_data/Y2(SO4)3_mp-16025_primitive.cif',r_en_limit,aaa,{{'Y' [89]} {'S' [33]} {'O' [17]}}, r_input, mat_input, z_direction);
        
        
        n_nu = r.number;									  %%number of nuclear spin site
        r_allall = r;
        r = r.all;		                    %% position of nuclear spin, electron spin is set at r=0
        i_nu = zeros(n_nu,1);
        mN = ones(n_nu,1)*mu_n;
        gN = zeros(n_nu,3);
        for k=1:n_nu
            i_nu(k,1) = ones(1,1)*nucspin(mat.all_spec{k});
            gN(k,:) = ones(1,3)*nucgval(mat.all_spec{k}); 					    %% (diagonal part of) gfactor tensor of nuclear spin
        end
        
    case 4 %(given)
        [r,mat] = load_crystal('given',r_en_limit,aaa,{{'C' [13]} }, r_input, mat_input, structdata.hdirection);
        n_nu = r.number;									  %%number of nuclear spin site
        r_allall = r;
        r = r.all;		                    %% position of nuclear spin, electron spin is set at r=0
        i_nu = [];
        mN = ones(n_nu,1)*mu_n;
        gN = zeros(n_nu,3);
        for k=1:n_nu
            i_nu(k,1) = ones(1,1)*nucspin(mat.all_spec{k});
            mN = ones(n_nu,1)*mu_n; 			    %% nuclear spin magneton
            gN(k,:) = ones(1,3)*nucgval(mat.all_spec{k}); 					    %% (diagonal part of) gfactor tensor of nuclear spin
        end
        
    otherwise
        warning('covfefe')
        return
end
end

