FROM debian:buster



RUN groupadd -g 9123 vcc && useradd -r -u 9123 -g vcc vcc && mkdir /home/vcc && chown vcc:vcc /home/vcc
WORKDIR /home/vcc

# Update and install python3, postgresql, compiler for sally
RUN apt-get update &&\ 
    apt-get install -y --no-install-recommends python3 python3-pip postgresql-11 postgresql-client-11 gcc libz-dev libconfig-dev libarchive-dev autoconf automake make libtool git xz-utils &&\
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install sally from github: https://github.com/rieck/sally
RUN git clone https://github.com/rieck/sally.git &&\
    cd sally && ./bootstrap && ./configure && make && make install

COPY ./ .
run chown vcc:vcc -R /home/vcc

USER root
# Get pip dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt


# Populate the postgresql DB, and run experiments
RUN service postgresql start &&\
    su - postgres -c "createuser vcc" &&\
    su - postgres -c "createdb vccfinder -O vcc" &&\
    su - postgres -c "echo CREATE EXTENSION IF NOT EXISTS hstore | psql vccfinder" &&\
    # Populate the DB \
    su - vcc -c "xzcat vccfinder-database.dump.xz | psql -v ON_ERROR_STOP=1 vccfinder"  &&\
    su - vcc -c "psql vccfinder -c 'CREATE INDEX IF NOT EXISTS idx_commits_id ON export.commits (id); CREATE INDEX IF NOT EXISTS idx_commits_blamed_id ON export.commits (blamed_commit_id); CREATE INDEX IF NOT EXISTS idx_cves_id ON export.cves (id); CREATE INDEX IF NOT EXISTS idx_repos_id ON export.repositories (id);' "

#RUN service postgresql start &&\
#    su - vcc -c "./do_everything.sh"



