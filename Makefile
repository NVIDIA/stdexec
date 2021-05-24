BUILDDIR = build

all: remote

setup:
	mkdir -p $(BUILDDIR)/

remote: setup
	find . -maxdepth 1 -name "*.bs" -type f | sed 's/\.bs$$//' | xargs -I{} -t -n 1 sh -c "curl https://api.csswg.org/bikeshed/ -F force=1 -F file=@{}.bs > $(BUILDDIR)/\`basename {}\`.html"

local: setup
	find . -maxdepth 1 -name "*.bs" -type f | sed 's/\.bs$$//' | xargs -I{} -t -n 1 sh -c "bikeshed -f spec {}.bs $(BUILDDIR)/\`basename {}\`.html"

clean:
	rm $(BUILDDIR)/*
	rmdir $(BUILDDIR)

