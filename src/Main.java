
import mpi.*;
import java.util.*;

public class Main {
    static final int ELEMENTS_PER_WORKER = 10;

    static final int TAG_DISTRIBUTE = 0;
    static final int TAG_WORKER = 1;
    static final int TAG_RESULT = 2;

    public static int[] merge(int[] a, int[] b) {
        int[] result = new int[a.length + b.length];

        int i = 0, j = 0, k = 0;

        while (i < a.length && j < b.length) {
            if (a[i] <= b[j]) {
                result[k++] = a[i++];
            } else {
                result[k++] = b[j++];
            }
        }

        while (i < a.length) {
            result[k++] = a[i++];
        }

        while (j < b.length) {
            result[k++] = b[j++];
        }

        return result;
    }

    public static void main(String[] args) throws Exception {
        MPI.Init(args);

        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();

        if (size < 4) {
            if (rank == 0) {
                System.out.println("Need at least 4 processes");
            }
            MPI.Finalize();
            return;
        }

        int workerCount = size - 3;

        MPI.COMM_WORLD.Barrier();
        long start = System.nanoTime();

        if (rank == 0) {
            int[] original =
                    new int[workerCount * ELEMENTS_PER_WORKER];

            Random random = new Random();

            for (int i = 0; i < original.length; i++) {
                original[i] = random.nextInt(1000);
            }

            System.out.println("Original array (" +
                    original.length + " elements):");
            System.out.println(Arrays.toString(original));

            int[][] sendBuffers =
                    new int[workerCount][ELEMENTS_PER_WORKER];

            Request[] sendRequests = new Request[workerCount];

            for (int i = 0; i < workerCount; i++) {
                System.arraycopy(
                        original,
                        i * ELEMENTS_PER_WORKER,
                        sendBuffers[i],
                        0,
                        ELEMENTS_PER_WORKER
                );

                sendRequests[i] = MPI.COMM_WORLD.Isend(
                        sendBuffers[i],
                        0,
                        ELEMENTS_PER_WORKER,
                        MPI.INT,
                        i + 3,
                        TAG_DISTRIBUTE
                );
            }

            Request.Waitall(sendRequests);

            int[][] received = new int[2][];

            for (int source = 1; source <= 2; source++) {
                Status status = MPI.COMM_WORLD.Probe(
                        source, TAG_RESULT
                );

                int count = status.Get_count(MPI.INT);

                received[source - 1] = new int[count];
            }

            Request[] receiveRequests = new Request[2];

            for (int source = 1; source <= 2; source++) {
                receiveRequests[source - 1] =
                        MPI.COMM_WORLD.Irecv(
                                received[source - 1],
                                0,
                                received[source - 1].length,
                                MPI.INT,
                                source,
                                TAG_RESULT
                        );
            }

            Request.Waitall(receiveRequests);

            System.out.println("\nProcess 1 sorted array (" +
                    received[0].length + " elements):");
            System.out.println(Arrays.toString(received[0]));

            System.out.println("\nProcess 2 sorted array (" +
                    received[1].length + " elements):");
            System.out.println(Arrays.toString(received[1]));

            long mergeStart = System.nanoTime();

            int[] result = merge(received[0], received[1]);

            long mergeTime = System.nanoTime() - mergeStart;

            System.out.println("\nFinal sorted array (" +
                    result.length + " elements):");
            System.out.println(Arrays.toString(result));

            System.out.println("\nMerging time: " +
                    mergeTime / 1_000_000.0 + " ms");

            System.out.println("Total time: " +
                    (System.nanoTime() - start) / 1_000_000.0 +
                    " ms");

            int[] expected = original.clone();
            Arrays.sort(expected);

            System.out.println("Sorting correct: " +
                    Arrays.equals(expected, result));

        } else if (rank == 1 || rank == 2) {
            List<Integer> sources = new ArrayList<>();

            for (int i = 3; i < size; i++) {
                int target = (i % 2 == 1) ? 1 : 2;

                if (target == rank) {
                    sources.add(i);
                }
            }

            int[] data =
                    new int[sources.size() * ELEMENTS_PER_WORKER];

            Request[] requests = new Request[sources.size()];

            for (int i = 0; i < sources.size(); i++) {
                requests[i] = MPI.COMM_WORLD.Irecv(
                        data,
                        i * ELEMENTS_PER_WORKER,
                        ELEMENTS_PER_WORKER,
                        MPI.INT,
                        sources.get(i),
                        TAG_WORKER
                );
            }

            if (requests.length > 0) {
                Request.Waitall(requests);
            }

            long sortStart = System.nanoTime();

            Arrays.sort(data);

            long sortTime = System.nanoTime() - sortStart;

            System.out.println("Process " + rank +
                    " sorting time: " +
                    sortTime / 1_000_000.0 + " ms");

            Request sendRequest = MPI.COMM_WORLD.Isend(
                    data,
                    0,
                    data.length,
                    MPI.INT,
                    0,
                    TAG_RESULT
            );

            sendRequest.Wait();

        } else {
            int[] data = new int[ELEMENTS_PER_WORKER];

            Request recvRequest = MPI.COMM_WORLD.Irecv(
                    data,
                    0,
                    ELEMENTS_PER_WORKER,
                    MPI.INT,
                    0,
                    TAG_DISTRIBUTE
            );

            recvRequest.Wait();

            int target = (rank % 2 == 1) ? 1 : 2;

            Request sendRequest = MPI.COMM_WORLD.Isend(
                    data,
                    0,
                    ELEMENTS_PER_WORKER,
                    MPI.INT,
                    target,
                    TAG_WORKER
            );

            sendRequest.Wait();
        }

        MPI.COMM_WORLD.Barrier();
        MPI.Finalize();
    }
}
