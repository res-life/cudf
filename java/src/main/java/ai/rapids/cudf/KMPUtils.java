/*
 *
 *  Copyright (c) 2024, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */
package ai.rapids.cudf;

import java.nio.charset.StandardCharsets;

/**
 * KMP utils for string-matching algorithm
 */
public class KMPUtils {

    /**
     * compute KMP next array for the UTF8 bytes of specified search pattern string.
     * @param searchPattern string search pattern will be used to search other strings.
     * @return KMP next array
     */
    public static int[] nextArray(String searchPattern) {
        byte[] patternBytes = searchPattern.getBytes(StandardCharsets.UTF_8);
        int[] next = new int[patternBytes.length];

        next[0] = -1;
        int j = 0;
        int k = -1;

        while (j < patternBytes.length - 1) {
            if (k == -1 || patternBytes[j] == patternBytes[k]) {
                j++;
                k++;
                next[j] = k;
            } else {
                k = next[k];
            }
        }

        return next;
    }
}
