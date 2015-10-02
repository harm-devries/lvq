from train_lvq import main

def jobman_entrypoint(state, channel):
    
    main(state=state, channel=channel, **state)

    return channel.COMPLETE

def jobman_insert_random(n_jobs):
    JOBDB = 'postgres://devries:12f19e7ecc@opter.iro.umontreal.ca/devries_db/lvq_mnist'
    EXPERIMENT_PATH = "lvq_mnist.jobman_entrypoint"

    jobs = []
    for _ in range(n_jobs):
        job = DD()

        job.n_hiddens = numpy.random.randint(1000, high=3000)
        job.n_out = numpy.random.randint(100, high=500)
        job.noise_std = numpy.random.uniform(low=0.0, high=0.8)
        job.learning_rate = 10.**numpy.random.uniform(-2, 0)
        job.momentum = 10.**numpy.random.uniform(-2, 0)
        job.gamma = numpy.random.uniform(low=1.0, high=3.0)
        
        job.tag = "lvq_mnist"

        jobs.append(job)
        print job

    answer = raw_input("Submit %d jobs?[y/N] " % len(jobs))
    if answer == "y":
        numpy.random.shuffle(jobs)

        db = jobman.sql.db(JOBDB)
        for job in jobs:
            job.update({jobman.sql.EXPERIMENT: EXPERIMENT_PATH})
            jobman.sql.insert_dict(job, db)

        print "inserted %d jobs" % len(jobs)
        print "To run: jobdispatch --condor --gpu --env=THEANO_FLAGS='floatX=float32, device=gpu' --repeat_jobs=%d jobman sql -n 1 'postgres://dauphiya:wt17se79@opter.iro.umontreal.ca/dauphiya_db/saddle_mnist_ae' ." % len(jobs)


def view(table="lvq_mnist",
         tag="dunno",
         user="devries",
         password="12f19e7ecc",
         database="devries_db",
         host="opter.iro.umontreal.ca"):
    """
    View all the jobs in the database.
    """
    import commands
    import sqlalchemy
    import psycopg2

    # Update view
    url = "postgres://%s:%s@%s/%s/" % (user, password, host, database)
    commands.getoutput("jobman sqlview %s%s %s_view" % (url, table, table))

    # Display output
    def connect():
        return psycopg2.connect(user=user, password=password,
                                database=database, host=host)

    engine = sqlalchemy.create_engine('postgres://', creator=connect)
    conn = engine.connect()
    experiments = sqlalchemy.Table('%s_view' % table,
                                   sqlalchemy.MetaData(engine), autoload=True)

    columns = [experiments.columns.id,
               experiments.columns.jobman_status,
               experiments.columns.tag,
               experiments.columns.nhiddens0,
               experiments.columns.learningrate,
               experiments.columns.momentum,
               experiments.columns.batchsize,
               experiments.columns.method,
               experiments.columns.trainerror,]

    results = sqlalchemy.select(columns,
                                order_by=[experiments.columns.tag,
                                    sqlalchemy.desc(experiments.columns.trainerror)]).execute()
    results = [map(lambda x: x.name, columns)] + list(results)

    def get_max_width(table, index):
        """Get the maximum width of the given column index"""
        return max([len(format_num(row[index])) for row in table])

    def format_num(num):
        """Format a number according to given places.
        Adds commas, etc. Will truncate floats into ints!"""
        try:
            if "." in num:
                return "%.7f" % float(num)
            else:
                return int(num)
        except (ValueError, TypeError):
            return str(num)

    col_paddings = []

    for i in range(len(results[0])):
        col_paddings.append(get_max_width(results, i))

    for row_num, row in enumerate(results):
        for i in range(len(row)):
            col = format_num(row[i]).ljust(col_paddings[i] + 2) + "|"
            print col,
        print

        if row_num == 0:
            for i in range(len(row)):
                print "".ljust(col_paddings[i] + 1, "-") + " +",
            print