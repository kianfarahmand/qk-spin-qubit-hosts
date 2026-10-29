% function [RotLeftMatrix] = LinTransToSetVectorTo001(mode);
% v = [1,-1,-1];
function [RotLeftMatrix] = LinTransToSetVectorTo001(v)

if prod(size(v) ~= [1,3] )|| (norm(v) == 0)
    error('rotation vector must have finite norm and size of [1,3]');
end

[phi,elevation,~] = cart2sph(v(1),v(2),v(3));
theta = pi / 2 - elevation;

S1 = [cos(phi), sin(phi),0; -sin(phi),cos(phi),0;0,0,1];
S2 = [cos(theta), 0, -sin(theta); 0,1,0; sin(theta), 0, cos(theta)];
RotLeftMatrix = S2 * S1;

% disp(phi * 180 / pi)
% disp(theta * 180 / pi)
% norm(v)
% S1 * v(:)
% RotLeftMatrix * v(:)

end

