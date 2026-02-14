#!/bin/bash
# List all PRISM4D releases and their status

echo "=== PRISM4D Release Status ==="
echo ""

echo "Git Tags:"
git tag -l | sort -V | while read tag; do
    date=$(git log -1 --format=%ai "$tag" 2>/dev/null | cut -d' ' -f1)
    msg=$(git tag -l -n1 "$tag" | cut -d' ' -f2-)
    echo "  $tag ($date) - $msg"
done

echo ""
echo "GitHub Releases:"
if command -v gh &> /dev/null; then
    gh release list --limit 20 2>/dev/null || echo "  (not connected to GitHub or no releases)"
else
    echo "  (gh CLI not installed)"
fi

echo ""
echo "Local Binaries:"
ls -la target/release/nhs-* target/release/generate-* target/release/cryptic-* 2>/dev/null | awk '{print "  " $NF " (" $5 " bytes)"}'
