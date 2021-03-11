for FILENAME in `ls -1 *pml`
do
    pymol -c $FILENAME
done
