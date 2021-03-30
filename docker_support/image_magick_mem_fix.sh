#!/bin/bash
#cat /etc/ImageMagick-6/policy.xml | grep '<policy domain="resource" name="memory"' -C5 | sed -e 's/<policy domain="resource" name="memory" value="[^"]\+"/<policy domain="resource" name="memory" value="5000MiB"/'
sed -e 's/<policy domain="resource" name="memory" value="[^"]\+"/<policy domain="resource" name="memory" value="5000MiB"/' -i /etc/ImageMagick-6/policy.xml
