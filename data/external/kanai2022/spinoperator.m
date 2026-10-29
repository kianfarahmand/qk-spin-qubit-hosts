%% Copyright (C) 2017 skanai
%% 
%% This program is free software; you can redistribute it and/or modify it
%% under the terms of the GNU General Public License as published by
%% the Free Software Foundation; either version 3 of the License, or
%% (at your option) any later version.
%% 
%% This program is distributed in the hope that it will be useful,
%% but WITHOUT ANY WARRANTY; without even the implied warranty of
%% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%% GNU General Public License for more details.
%% 
%% You should have received a copy of the GNU General Public License
%% along with this program.  If not, see <http://www.gnu.org/licenses/>.

%% -*- texinfo -*- 
%% @deftypefn {} {@var{retval} =} spinoperator (@var{input1}, @var{input2})
%%
%% @seealso{}
%% @end deftypefn

%% Author: skanai <skanai@SPINACH>
%% Created: 2017-06-11
%% S(1): Sx
%% S(2): Sy
%% S(3): Sz
%% S(4): S+
%% S(5): S-
%% S(6): S(S+1)

function [S] = spinoperator (i_sp)
  %%S = zeros(2*n_sp+1);
  s = i_sp*(i_sp+1);
  S = complex(zeros(2*i_sp+1,2*i_sp+1,6));
  for m=1:2*i_sp+1;
    for n=1:2*i_sp+1;
      S(m,n,1) = (kronDel(m,n+1)+kronDel(m+1,n))/2*sqrt(s-(i_sp-m+1)*(i_sp-n+1));
      S(m,n,2) = (-kronDel(m,n+1)+kronDel(m+1,n))/2/i*sqrt(s-(i_sp-m+1)*(i_sp-n+1));
      S(m,n,3) = kronDel(m,n)*(i_sp-m+1);
      S(m,n,4) = kronDel(m+1,n)*sqrt(s-(i_sp-m+1)*(i_sp-n+1));
      S(m,n,5) = kronDel(m,n+1)*sqrt(s-(i_sp-m+1)*(i_sp-n+1));
      S(m,n,6) = kronDel(m,n)*s;
    end
  end
end