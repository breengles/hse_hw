#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long

int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);

    assert(freopen("editdist.in", "r", stdin));
    assert(freopen("editdist.out ", "w", stdout));
    
    string l1, l2;
    cin >> l1 >> l2;
    int n = l1.length();
    int m = l2.length();

    int f[n+1][m+1];

    for (int i = 0; i <= n; i++){
        for (int j = 0; j <= m; j++){
            if (i == 0){
                f[i][j] = j;
            }
            else if (j == 0){
                f[i][j] = i;
            }
            else if (l1[i - 1] == l2[j - 1]){
                f[i][j] = f[i - 1][j - 1];
            }
            else{
                f[i][j] = 1 + min(min(f[i][j - 1], f[i - 1][j]), f[i - 1][j - 1]);
            }
        }
    }
    cout << f[n][m];
    return 0;
}