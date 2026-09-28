function h5create(file, ds, sz, varargin)
% Test shim for GNU Octave (no h5create): keeps datasets in memory; see h5shim_dump.m.
global H5SHIM
if isempty(H5SHIM), H5SHIM = containers.Map(); end
k = [file '|' ds];
dt = 'double';
for i = 1:2:numel(varargin), if strcmpi(varargin{i}, 'Datatype'), dt = varargin{i+1}; end, end
if isscalar(sz), sz = [sz 1]; end
H5SHIM(k) = zeros(sz, dt);
end
