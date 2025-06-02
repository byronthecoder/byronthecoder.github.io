#!/bin/bash

# Find all post files with "categories: sample-posts"
grep -l "categories: sample-posts" _posts/*.md | while read file; do
  # Check if "published: false" is already in the file
  if ! grep -q "published: false" "$file"; then
    # Add "published: false" after the categories line
    sed -i '' 's/categories: sample-posts/categories: sample-posts\npublished: false/' "$file"
    echo "Updated $file to be unpublished"
  fi
done

echo "All sample posts have been marked as unpublished"
