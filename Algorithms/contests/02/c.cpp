#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long

int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);
    
    assert(freopen("lcs.in", "r", stdin));
    assert(freopen("lcs.out", "w", stdout));

    int n;
    cin >> n;

    int s1[n];
    for (int i = 0; i < n; i++){
        cin >> s1[i];
    }

    int m;
    cin >> m;

    int s2[m];
    for (int i = 0; i < m; i++){
        cin >> s2[i];
    }

    int d[m + 1][n + 1];
    for (int i = 0; i <= m; i++){
        for (int j = 0; j <= n; j++){
            if (i == 0 || j == 0){
                d[i][j] = 0;
            }
            else if (s1[j - 1] == s2[i - 1]){
                d[i][j] = d[i - 1][j - 1] + 1;
            }
            else{
                d[i][j] = max(d[i - 1][j], d[i][j - 1]);
            }
        }
    }

    cout << d[m][n];


    return 0;
}