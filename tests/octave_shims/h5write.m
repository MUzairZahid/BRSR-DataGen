function h5write(file, ds, data, start, count)
global H5SHIM
k = [file '|' ds];
A = H5SHIM(k);
if isvector(A) && numel(start) == 1
    assert(start + count - 1 <= numel(A), 'out of range'); A(start:start+count-1) = data(:);
else
    idx = arrayfun(@(s, c) s:s+c-1, start, count, 'UniformOutput', false);
    for d = 1:numel(start), assert(start(d) + count(d) - 1 <= size(A, d), 'out of range dim %d', d); end
    assert(isequal(size(data, 1:numel(count)), count), 'block size mismatch');
    A(idx{:}) = data;
end
H5SHIM(k) = A;
end
