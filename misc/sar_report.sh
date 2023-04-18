#!/bin/bash
LC_ALL=C sar -r > $HOME/sar_report/$(date +%d)_$(date +%b).txt
