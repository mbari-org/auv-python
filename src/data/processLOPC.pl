#!/usr/bin/perl -w
#
# Script to loop through lopc.bin files and run lopcToNetCDF.py on them
# to replicate what is done by the auvportal for the standard processing
# of dorado mission log files.
#
# Use SSDS.pm to submit processing metadata to SSDS_Metadata.
#
# Mike McCann
# 3 March 2010
#
# $Id: processLOPC.pl,v 1.7 2010/09/09 16:09:58 ssdsadmin Exp $

use SSDS;

sub usage() {
	print "Usage: $0 YYYY | mission \n\nWhere YYYY is between 2006 and the current year or mission is of form YYYYY.DDD.NN\n\n";
	exit(0)
} 

# Very short test mission
# $binFile = '/mbari/AUVCTD/missionlogs/2009/2009084/2009.084.00/lopc.bin';
# $ncFile = '/mbari/ssdsdata/ssds/generated/netcdf/files/ssds.shore.mbari.org/auvctd/missionlogs/2009/2009084/2009.084.00/lopc.nc';
# $logFile = '/mbari/ssdsdata/ssds/generated/netcdf/files/ssds.shore.mbari.org/auvctd/missionlogs/2009/2009084/2009.084.00/lopc.nc.log';

# A test mission that displays the sampleCount overflow problem - takes about 20 minutes to run
##$binFile = '/mbari/AUVCTD/missionlogs/2009/2009124/2009.124.03/lopc.bin';
##$ncFile = '/mbari/ssdsdata/ssds/generated/netcdf/files/ssds.shore.mbari.org/auvctd/missionlogs/2009/2009124/2009.124.03/lopc.nc';
##$logFile = '/mbari/ssdsdata/ssds/generated/netcdf/files/ssds.shore.mbari.org/auvctd/missionlogs/2009/2009124/2009.124.03/lopc.nc.log';
##system("lopcToNetCDF.py -i $binFile -n $ncFile -v -f > $logFile 2>&1");
##exit;


$arg = $ARGV[0];

print "arg = $arg\n";
usage() unless $arg;

##-die "Usage: $0 YYYY | mission \n\nWhere YYYY is between 2006 and the current year or mission is of form YYYYY.DDD.NN\n\n" unless ($arg > 2002 || $arg =~ /\d\d\d\d\.\d\d\d\.\d\d/);

# For mainly setting transparency and Large Copepod criteria consistently
$extraArgs = " --LargeCopepod_AIcrit 0.3";

if ( $arg =~ /^\d\d\d\d$/ ) {
	$YYYY = $arg;
	usage() if $arg < 2006;
	
	# Process all missions within the year
	$findCmd = "find /mbari/AUVCTD/missionlogs/$YYYY -name lopc.bin -print";
	print "Executing command: \n$findCmd\n";
	open (FIND, "$findCmd |");
	while (<FIND>) {

        	##print $_;
		chop($binFile = $_);
		if ($binFile =~ /mbari\/AUVCTD\/(.+)\/lopc.bin/ ) {
			$mPath = $1;	
		}
		##die $mPath;
		$ncFile = "/mbari/ssdsdata/ssds/generated/netcdf/files/ssds.shore.mbari.org/auvctd/${mPath}/lopc.nc";
		$logFile = "${ncFile}.log";

		$cmd = "lopcToNetCDF.py -i $binFile -n $ncFile -v -f $extraArgs 2>&1 | tee $logFile";
	
		print "$cmd\n";
		system($cmd)
	} # End while(<FIND>)

}
elsif ( $arg =~ /(\d\d\d\d)\.(\d\d\d)\.(\d\d)/ ) {
	$mission_name = $arg;
	$YYYY = $1;
	$DDD = $2;

	$binFile = "/mbari/AUVCTD/missionlogs/$YYYY/${YYYY}${DDD}/$mission_name/lopc.bin";
	$ncFile = "/mbari/ssdsdata/ssds/generated/netcdf/files/ssds.shore.mbari.org/auvctd/missionlogs/$YYYY/${YYYY}${DDD}/$mission_name/lopc.nc";
	$logFile = "/mbari/ssdsdata/ssds/generated/netcdf/files/ssds.shore.mbari.org/auvctd/missionlogs/$YYYY/${YYYY}${DDD}/$mission_name/lopc.nc.log";
	$cmd = "lopcToNetCDF.py -i $binFile -n $ncFile -v -f $extraArgs 2>&1 | tee $logFile";
	print "Executing: $cmd\n";
	system($cmd);

}
else {
	usage();

} # End if ( $arg =~ /\d\d\d\d/ ) 

