#!/usr/bin/env perl

# deletes the sentences with more than 50% english words
# usage: ./detect_en_sentence.pl < in > out
# author: Bushra Jawaid <jawaid@ufal.mff.cuni.cz>

use strict;
use warnings;
use utf8;

binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";

my $sentence_no =1;

while (<>) {
   my @tokens = split / /, $_;
   my $matched = () = /[a-zA-Z]+/g;
   print $_ if ($matched/scalar(@tokens) <= 0.5);
}
