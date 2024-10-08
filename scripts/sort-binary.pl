use strict;
use warnings;

# Open the binary file for reading
open(my $infile, '<:raw', 'txt/rmat-16-16.bin') or die "Cannot open input.bin: $!";
# Open the binary file for writing the sorted output
open(my $outfile, '>:raw', 'output.bin') or die "Cannot open output.bin: $!";

my @data;

# Read the binary file 16 bytes at a time
while (read($infile, my $record, 16)) {
    my ($src, $dst) = unpack('Q<Q<', $record);  # Unpack two 64-bit unsigned integers (src, dst)
    push @data, { src => $src, dst => $dst, record => $record };  # Store the record for sorting
}

# Sort the records by the 'src' field
@data = sort { $a->{src} <=> $b->{src} } @data;

# Write the sorted records back to the output file
for my $entry (@data) {
    print $outfile $entry->{record};  # Write the original 16-byte record
}

# Close file handles
close($infile);
close($outfile);

print "Sorting complete. Output written to output.bin\n";
