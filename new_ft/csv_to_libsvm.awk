#!/usr/bin/awk -f

# Super simple CSV to Libsvm transformation
# This script REQUIRES a LABEL parameter to be passed
# Call this script with -v LABEL="0" or -v LABEL="1"
BEGIN {
	FS=","; 
	OFS=" ";
	if (length(LABEL) != 1) {print "csv_tolibsvm.awk: no label provided, aborting" > "/dev/stderr"; exit 1}
}
	# In the CSV, id is in first col
	# We want it as a comment at the end of a libsvm line
	#  -> Print all fields AFTER the first one, then wrote the first one as a comment
	{ 
		printf "%s ", LABEL; #in libsvm, label is at the begining if the line
		for(i=2;i<=NF;i++) # all features (starting at 2 to discard id)
		{
			printf "%s:%s ", (i-1), $i
		}; 
		printf "#%s", $1 ;
		printf "\n";
	}
