#!/usr/bin/env bash
# One literal mutation, one focused test-run, byte-identical restoration.
# Usage: tools/mutation-run.sh <module> <patch-file> <@alias>
# Patch bytes are OLD@@@NEW (exactly one delimiter; no newline is stripped).
# OLD must occur exactly once, including overlapping occurrences. NEW may be empty.
# Use an otherwise idle, isolated worktree; do not edit the module during the run.
# Exit: test-run's status, 2 for refusal, 3 for failed restoration; signals 128+N.
# INT/TERM/HUP cancel and reap test-run before restoration. SIGKILL/power loss
# cannot be trapped: the printed recovery copy is retained until cmp succeeds.
exec perl - "$0" "$@" <<'PERL'
use strict;
use warnings;
use Cwd qw(abs_path);
use File::Basename qw(dirname basename);
use File::Temp qw(tempdir tempfile);
use File::Copy qw(copy);
use Errno qw(EINTR);

my $script = shift;
$SIG{__DIE__} = sub { if (!$^S) { print STDERR @_; exit 2; } };
sub refuse { die "mutation-run: $_[0]\n" }
@ARGV == 3 or refuse('usage: tools/mutation-run.sh <module> <patch-file> <@alias>');
my ($module, $patch, $alias) = @ARGV;
my $root = abs_path(dirname($script) . '/..');
$alias =~ /^\@[^\s]+$/ or refuse('expected one @alias');
-f $module && !-l $module or refuse('module must be a regular, non-symlink file');
$module = abs_path($module);
index($module, "$root/") == 0 or refuse('module must be inside this worktree');
sub bytes {
    open my $f, '<:raw', $_[0] or refuse("read $_[0]: $!");
    local $/;
    my $b = <$f>;
    close $f or refuse("close $_[0]: $!");
    return $b;
}
my $original = bytes($module);
my @parts = split /\@\@\@/, bytes($patch), -1;
@parts == 2 && length($parts[0]) or refuse('patch needs one @@@ delimiter and a nonempty OLD');
my ($old, $new) = @parts;
my $at = index($original, $old);
$at >= 0 or refuse('missing anchor');
index($original, $old, $at + 1) < 0 or refuse('ambiguous anchor');
$old ne $new or refuse('mutation does not change the module');
my $mutated = $original;
substr($mutated, $at, length($old), $new);
my $scratch = tempdir('ocannl-mutation-XXXXXXXX', TMPDIR => 1, CLEANUP => 0);
my $backup = "$scratch/original";
copy($module, $backup) or refuse("backup: $!");
my $mode = (stat($module))[2] & 07777;
chmod($mode, $backup) or refuse("backup permissions: $!");
$| = 1;
print "recovery: $backup\n";
my ($child, $cancel, $changed) = (0, 0, 0);
for my $pair ([INT => 2], [TERM => 15], [HUP => 1]) {
    my ($name, $number) = @$pair;
    $SIG{$name} = sub { $cancel ||= 128 + $number; kill 'TERM', $child if $child; };
}
my $code = 2;
my $error;
my $report = "";
eval {
    chdir $root or refuse("chdir: $!");
    refuse('cancelled before mutation') if $cancel;
    # Mark before opening: even a short write must restore the complete backup.
    $changed = 1;
    open my $f, '>:raw', $module or refuse("write module: $!");
    print {$f} $mutated or refuse("write module: $!");
    close $f or refuse("close module: $!");
    refuse('cancelled before launch') if $cancel;
    $child = fork();
    defined $child or refuse("fork: $!");
    if (!$child) {
        $SIG{$_} = 'DEFAULT' for qw(INT TERM HUP);
        # A fork child must never unwind into the parent's restoration block.
        open STDIN, '<', '/dev/null' or exit 126;
        open STDOUT, '>', "$scratch/transcript" or exit 126;
        open STDERR, '>&', \*STDOUT or exit 126;
        exec('bash', 'tools/test-run.sh', 'run', 'build', '-j', '4', $alias) or do {
            print STDERR "exec test-run: $!\n";
            exit 127;
        };
    }
    kill 'TERM', $child if $cancel;
    while (1) {
        my $got = waitpid($child, 0);
        next if $got < 0 && $! == EINTR;
        $got == $child or refuse("waitpid: $!");
        $code = ($? & 127) ? 128 + ($? & 127) : $? >> 8;
        $child = 0;
        last;
    }
    my $transcript = bytes("$scratch/transcript");
    $report = $transcript;
    # Use this invocation's digest, never the mutable 'last' pointer or its
    # abbreviated log tail (which can repeat or omit false claims).
    if ($transcript =~ /^log: +(.+\/log)\r?$/m) {
        my $log = $1;
        $log =~ s/\r$//;
        $report .= 'run: ' . basename(dirname($log)) . "\nfalse claims:\n";
        my @claims = bytes($log) =~ /^(FAIL: .*: false)\r?$/mg;
        $report .= "$_\n" for @claims;
        $report .= "(none)\n" unless @claims;
    } else {
        $report .= "run: unavailable (test-run produced no digest)\n";
        $code = 2 unless $code;
    }
    1;
} or $error = $@;
# No interrupt may cut the restoration in half. Publish a complete replacement
# in the same directory, then independently compare against the recovery copy.
$SIG{$_} = 'IGNORE' for qw(INT TERM HUP);
if ($changed) {
    my $restored = eval {
    my ($restore_fh, $restore) = tempfile('.mutation-restore-XXXXXXXX', DIR => dirname($module), UNLINK => 0);
    close $restore_fh;
    copy($backup, $restore) && chmod($mode, $restore) && rename($restore, $module)
            && system('cmp', '-s', $backup, $module) == 0;
    };
    unless ($restored) {
        print STDERR "mutation-run: RESTORATION FAILED; recover from $backup ($!)\n";
        exit 3;
    }
    $report .= "restored: byte-identical (cmp)\n";
}
unlink $backup;
# Publish only after restoration, so a closed output pipe cannot strand a mutant.
print $report;
print STDERR $error if $error;
exit($cancel || ($error ? 2 : $code));
PERL
