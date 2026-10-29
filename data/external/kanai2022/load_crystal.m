%% Author: skanai <skanai@SPINACH>
%% Created: 2017-06-14

function [r, mat] = load_crystal (structure, cut_off, lattice_constant, materials, r_input, mat_input, H_direction)

%% rotate bath so that external magnetic field direction is set to [001]
% for 'given' case, no transformation
[RotLeftMatrix] = LinTransToSetVectorTo001(H_direction);

%% structure;

mat.num = [];
switch structure
    case 'sc'
        r.base = (RotLeftMatrix * [1,0,0;0,1,0;0,0,1]')' * lattice_constant;
        r.index_number_mat = 1;
        r.index_number_spec = 1;
        r.index(1,:) = [0,0,0] * lattice_constant;%% r.index(x,y,z,points of r in unit cell,species)
        r.index = (RotLeftMatrix * r.index')';
        mat.num = 1;
        %         mat.num = [1];
        mat.name{1} = materials{1}{1};
        r.volumeperatom = dot((cross(r.base(1,:), r.base(2,:))),r.base(3,:)) / r.index_number_mat;
        
    case 'bcc'
        r.base = (RotLeftMatrix * [1,0,0;0,1,0;0,0,1]')' * lattice_constant;
        r.index_number_mat = 2;
        r.index_number_spec = 1;
        r.index = zeros(r.index_number_mat,3); %% r in unit cell, (x,y,z), species%
        r.index(1,:) = [0,0,0]*lattice_constant;%% r.index(x,y,z,points of r in unit cell,species)
        r.index(2,:) = [1/2,1/2,1/2]*lattice_constant;%% r.index(x,y,z,points of r in unit cell,species)
        r.index = (RotLeftMatrix * r.index')';
        mat.num = [1,1];
        mat.name{1} = materials{1}{1};
        r.volumeperatom = dot((cross(r.base(1,:), r.base(2,:))),r.base(3,:)) / r.index_number_mat;
        
    case 'fcc'
        r.base = (RotLeftMatrix * [1,0,0;0,1,0;0,0,1]')' * lattice_constant;
        r.index_number_mat = 4;
        r.index_number_spec = 1;
        r.index = zeros(r.index_number_mat,3); %% r in unit cell, (x,y,z), species%
        r.index(1,:) = [0,0,0]*lattice_constant;%% r.index(x,y,z,points of r in unit cell,species)
        r.index(2,:) = [0,1/2,1/2]*lattice_constant;%% r.index(x,y,z,points of r in unit cell,species)
        r.index(3,:) = [1/2,0,1/2]*lattice_constant;%% r.index(x,y,z,points of r in unit cell,species)
        r.index(4,:) = [1/2,1/2,0]*lattice_constant;%% r.index(x,y,z,points of r in unit cell,species)
        r.index = (RotLeftMatrix * r.index')';
        mat.num = [1,1,1,1];
        mat.name{1} = materials{1}{1};
        r.volumeperatom = dot((cross(r.base(1,:), r.base(2,:))),r.base(3,:)) / r.index_number_mat;
        
    case 'hcp'
        r.base = (RotLeftMatrix * [1, 0, 0; 0.5, 3^0.5/2, 0; 0, 0, 2 * (2/3)^0.5]')' * lattice_constant;
        r.index_number_mat = 2;
        r.index_number_spec = 1;
        r.index = zeros(r.index_number_mat,3); %% r in unit cell, (x,y,z), species%
        r.index(1,:) = [0,0,0]*lattice_constant;%% r.index(x,y,z,points of r in unit cell,species)
        r.index(2,:) = [1.5, 3^0.5/2, (2/3)^0.5]/3*lattice_constant;%% r.index(x,y,z,points of r in unit cell,species)
        r.index = (RotLeftMatrix * r.index')';
        mat.num = [1,1];
        mat.name{1} = materials{1}{1};
        r.volumeperatom = dot((cross(r.base(1,:), r.base(2,:))),r.base(3,:)) / r.index_number_mat;
        
    case 'random'
        boxnum = 100;
        lattice_constant = lattice_constant * ((boxnum)^(1/3));
        r.base = (RotLeftMatrix * [1,0,0;0,1,0;0,0,1]')' * lattice_constant ;
        r.index_number_mat = boxnum;
        r.index_number_spec = 1;
        %         r.index = rand(boxnum,3) * lattice_constant;%% r.index(x,y,z,points of r in unit cell,species)
        
        r.index = rand(1,3) * lattice_constant;
        for i = 1:boxnum-1
            while 1
                a = rand(1,3) * lattice_constant;
                if min(vecnorm((r.index-a)')) > 0.8e-10 %カットオフ以下ならもう一度やる
                    r.index = [r.index;a];
                    break
                end
            end
        end
        
        r.index = (RotLeftMatrix * r.index')';
        mat.num = ones(1,boxnum);
        mat.name{1} = materials{1}{1};
        r.volumeperatom = dot((cross(r.base(1,:), r.base(2,:))),r.base(3,:)) / r.index_number_mat;
        
    case 'diamond' %OK
        r.base = (RotLeftMatrix * [1,0,0;0,1,0;0,0,1]')' * lattice_constant;
        r.index_number_mat = 8;
        r.index_number_spec = 1;
        r.index = zeros(r.index_number_mat,3); %% r in unit cell, (x,y,z), species%
        r.index(1,:) = [0,0,0]*lattice_constant;%% r.index(x,y,z,points of r in unit cell,species)
        r.index(2,:) = [1/2,1/2,0]*lattice_constant;%% r.index(x,y,z,points of r in unit cell,species)
        r.index(3,:) = [1/2,0,1/2]*lattice_constant;%% r.index(x,y,z,points of r in unit cell,species)
        r.index(4,:) = [0,1/2,1/2]*lattice_constant;%% r.index(x,y,z,points of r in unit cell,species)
        r.index(5,:) = [1/4,1/4,1/4]*lattice_constant;%% r.index(x,y,z,points of r in unit cell,species)
        r.index(6,:) = [3/4,3/4,1/4]*lattice_constant;%% r.index(x,y,z,points of r in unit cell,species)
        r.index(7,:) = [3/4,1/4,3/4]*lattice_constant;%% r.index(x,y,z,points of r in unit cell,species)
        r.index(8,:) = [1/4,3/4,3/4]*lattice_constant;%% r.index(x,y,z,points of r in unit cell,species)
        r.index = (RotLeftMatrix * r.index')';
        mat.num = [1,1,1,1,1,1,1,1];
        mat.name{1} = materials{1}{1};
        r.volumeperatom = dot((cross(r.base(1,:), r.base(2,:))),r.base(3,:)) / r.index_number_mat;
        
        %         r.base = (RotLeftMatrix * [0.252761893,0,0; 0.1263809465,0.218898220446644,0; 0.1263809465, 0.072966073482215,0.206379221423320]')' / 0.357 * lattice_constant;
        %         r.index_number_mat = 2;
        %         r.index_number_spec = 1;
        %         r.index = [0.252761893,0.145932146964429,0.103189610711660; 0.3791428395,0.218898220446644,0.154784416067490] / 0.357 * lattice_constant;
        %         r.index = (RotLeftMatrix * r.index')';
        %         mat.num = [1,1];
        %         mat.name{1} = materials{1}{1};
        %
        %         r.volumeperatom = dot((cross(r.base(1,:), r.base(2,:))),r.base(3,:)) / r.index_number_mat;
        
    case 'given'
        r = r_input;
        mat = mat_input;
        
    otherwise
        [temp_a]    = textread(structure,'%s');
        temp_b = char(temp_a);
        temp_b_str = cellstr(temp_b);
        
        temp_c = strcmp(temp_a,'_cell_length_a');
        cell_length_a = str2double(temp_a(find(temp_c>0)+1));
        
        temp_c = strcmp(temp_a,'_cell_length_b');
        cell_length_b = str2double(temp_a(find(temp_c>0)+1));
        
        temp_c = strcmp(temp_a,'_cell_length_c');
        cell_length_c = str2double(temp_a(find(temp_c>0)+1));
        
        temp_c = strcmp(temp_a,'_cell_angle_alpha');
        cell_angle_alpha = str2double(temp_a(find(temp_c>0)+1));
        
        temp_c = strcmp(temp_a,'_cell_angle_beta');
        cell_angle_beta = str2double(temp_a(find(temp_c>0)+1));
        
        temp_c = strcmp(temp_a,'_cell_angle_gamma');
        cell_angle_gamma = str2double(temp_a(find(temp_c>0)+1));
        
        %% put r.base here
        r.base(1,:) = [1,0,0] * cell_length_a * 1e-10;
        r.base(2,:) = [cos(cell_angle_gamma*pi/180),sin(cell_angle_gamma*pi/180),0] * cell_length_b * 1e-10;
        c_1 = cos(cell_angle_beta*pi/180);
        c_2 = (cos(cell_angle_alpha*pi/180)-cos(cell_angle_beta*pi/180)*cos(cell_angle_gamma*pi/180))/(sin((cell_angle_gamma*pi/180)));
        c_3 = sqrt(1-c_1^2-c_2^2);
        r.base(3,:) = [c_1, c_2, c_3] * cell_length_c * 1e-10;
        r.base = (RotLeftMatrix * r.base')';
        
        temp_c = strcmp(temp_a,'_atom_site_occupancy');
        temp_d = find(temp_c>0);
        
        r.index_number_spec = size(materials,2);
        for m =1:r.index_number_spec
            mat.name{m} = materials{m}{1};
        end
        
        l = 1;
        for k = 1:1e5
            if temp_d+1+(l-1)*7 > size(temp_a,1)
                break
            else
                temp_mat = temp_b_str(temp_d + 1 + (l-1)*7);
                for o=1:size(materials,2)
                    if strcmp(temp_mat,materials{o}{1})
                        mat.num(l) = o;
                        break
                    end
                end
                r.index_temp(l,:) = [str2double(temp_a(temp_d+(l-1)*7+4)),str2double(temp_a(temp_d+(l-1)*7+5)),str2double(temp_a(temp_d+(l-1)*7+6))];
                l = l + 1;
            end
        end
        r.index_number_mat = l - 1;
        
        for k = 1:r.index_number_mat
            r.index(k,:) = r.base(1,:) * r.index_temp(k,1) + r.base(2,:) * r.index_temp(k,2) + r.base(3,:) * r.index_temp(k,3);
        end
        r.index = (RotLeftMatrix * r.index')';
        
                r.volumeperatom = dot((cross(r.base(1,:), r.base(2,:))),r.base(3,:)) / r.index_number_mat;
%         r.volumeperatom = (cross(r.base(1,:), r.base(2,:))) * r.base(3,:) / r.index_number_mat;
end


%% set species
switch structure
    case 'given'
        
    otherwise
        m = 0;
        for k= 1:r.index_number_spec
            for l = 1:size(materials{k}{2},2)
                m = m + 1;
                mat.spec{m} = strcat(num2str(materials{k}{2}(l)),materials{k}{1});
            end
        end
        r.index_number_spec_iso = m;
        
        for k=1:r.index_number_spec
            if size(materials{k},2) < 3
                for l = 1:size(materials{k}{2},2)
                    materials{k}{3}(l) = nucabund(strcat(num2str(materials{k}{2}(l)),materials{k}{1}));
                end
            end
        end
        
        
        %% set basis  %%%%%%%%%%%%%%%%%%%%%%
        k2 = 0;
        r.all = [];
        mat.all = [];
        for l = -ceil(cut_off/norm(r.base(1,1:3))):ceil(cut_off/norm(r.base(1,1:3)))
            for m = -ceil(cut_off/norm(r.base(2,1:3))):ceil(cut_off/norm(r.base(2,1:3)))
                for n = -ceil(cut_off/norm(r.base(3,1:3))):ceil(cut_off/norm(r.base(3,1:3)))
                    for k1 = 1:r.index_number_mat
                        temp_dice = rand();
                        temp_probability = materials{mat.num(k1)}{3}(1);
                        for k3 = 1:size(materials{mat.num(k1)}{2},2)
                            if (temp_dice < temp_probability)
                                k2 = k2 + 1;
                                R0 = l*r.base(1,1:3)+m*r.base(2,1:3)+n*r.base(3,1:3);
                                r.all(k2,1:3) = R0 + r.index(k1,1:3);
                                mat.all(k2) = mat.num(k1);
                                mat.all_spec{k2} = strcat(num2str(materials{mat.num(k1)}{2}(k3)),materials{mat.num(k1)}{1});
                            elseif k3 == size(materials{mat.num(k1)}{2},2)
                                break
                            else
                                temp_probability = temp_probability + materials{mat.num(k1)}{3}(k3+1);
                            end
                        end
                    end
                end
            end
        end
        r.number = k2;
end
%% erase outside of cut_off
k4 = 0;
for k3 = 1 : r.number
    if ((0.4e-10 > norm(r.all(k3-k4,1:3))) || (norm(r.all(k3-k4,1:3)) > cut_off) || (nucspin(mat.all_spec{k3}) == 0))
        r.all(k3-k4,:) = [];
        mat.all(k3-k4) =[];
        mat.all_spec{k3} = [];
        k4 = k4 + 1;
    end
end

r.number = r.number - k4;
if r.number > 0
    x = find(cellfun('isempty',mat.all_spec)); % ???????
    mat.all_spec(x) = []; % ?????????
end

end
