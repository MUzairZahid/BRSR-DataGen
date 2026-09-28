function h5shim_dump(out_mat)
% Save all shimmed datasets to a MAT file (-v7) for inspection from Python.
global H5SHIM
S = struct(); ks = keys(H5SHIM);
for i = 1:numel(ks)
    [f, ds] = strtok(ks{i}, '|'); [~, name] = fileparts(f);
    S.([name '__' ds(3:end)]) = H5SHIM(ks{i});
end
save('-v7', out_mat, '-struct', 'S');
end
