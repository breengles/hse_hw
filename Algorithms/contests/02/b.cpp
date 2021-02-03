#include <bits/stdc++.h>

using namespace std;

#define ui unsigned int
#define ll long long
#define ul unsigned long
#define ull unsigned long long


int max(int a, int b){ 
    return (a > b) ? a : b; 
} 

int main(){
    ios_base::sync_with_stdio(0);
    cin.tie(NULL);

    assert(freopen("knapsack.in", "r", stdin));
    assert(freopen("knapsack.out ", "w", stdout));

    int n, s;
    cin >> s >> n;
    int w[n], sum = 0;
    for (int i = 0; i < n; i++){
        cin >> w[i];
        sum += w[i];
    }
    if (sum <= s){
        cout << sum;
        return 0;
    }

    int d[s + 1][n + 1];
    for (int i = 0; i <= n; i++) { 
        for (int j = 0; j <= s; j++) { 
            if (i == 0 || j == 0) 
                d[j][i] = 0; 
            else if (w[i - 1] <= j)
                d[j][i] = max(d[j][i - 1], w[i - 1] + d[j - w[i - 1]][i - 1]); 
            else
                d[j][i] = d[j][i - 1]; 
        } 
    } 
  
    cout << d[s][n];

    return 0;
}