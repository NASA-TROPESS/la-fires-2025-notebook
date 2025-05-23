module utils

function check_mtime(output_file, input_files)
    if !isfile(output_file)
        @info "Output file $output_file does not exist, must create"
        return true
    end

    output_mtime = mtime(output_file)
    for file in input_files
        input_mtime = mtime(file)
        if input_mtime > output_mtime
            @info "Input file $file is newer than $output_file, must recreate output file"
            return true
        end
    end
    @info "Output file $output_file is newer than all inputs, no need to recreate"
    return false

end

function isclose(a, b)
    all(abs.(a - b) .< 1e-8)
end

end
