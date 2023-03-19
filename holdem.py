import numpy as np

MIN_BET = 1

a = []
for i in range(52):
    a.append([i // 4, i % 4])
CARD_DICT = dict(enumerate(a))

PROCESS = ['blind', 'pre_flop', 'pre_flop_bet', 'flop', 'flop_bet', 'turn', 'turn_bet', 'river', 'river_bet', 'settle']
NONE_BET = ['blind', 'pre_flop', 'flop', 'turn', 'river', 'settle', 'settle']
BET = ['pre_flop_bet', 'flop_bet', 'turn_bet', 'river_bet']


class Deck:
    def __init__(self):
        self.deck = [0 for _ in range(52)]
        self.comm = [0 for _ in range(52)]

    def dispatch(self):
        rand = np.random.randint(low=0, high=52, size=1)[0]
        if self.deck[rand] == 1:
            while self.deck[rand] == 1:
                rand = np.random.randint(low=0, high=52, size=1)[0]

        self.deck[rand] = 1
        return rand


class Player:
    def __init__(self, money):
        self.hand = [0 for _ in range(52)]
        self.hand_comm = [0 for _ in range(52)]
        self.hc = None
        self.hc_show = None
        self.money = money

    def renew_handcomm(self, comm):
        for idx, card in enumerate(comm):
            if card == 1:
                self.hand_comm[idx] = 1
        hc = []
        hcd = []
        for idx, card in enumerate(self.hand_comm):
            if card == 1:
                temp = CARD_DICT[idx]
                hc.append(temp)
                if temp[1] == 0:
                    suit = '♥'
                elif temp[1] == 1:
                    suit = '♦'
                elif temp[1] == 2:
                    suit = '♣'
                elif temp[1] == 3:
                    suit = '♠'
                if temp[0] == 0 or temp[0] == 13:
                    num = 'A'
                elif temp[0] == 10:
                    num = 'J'
                elif temp[0] == 11:
                    num = 'Q'
                elif temp[0] == 12:
                    num = 'K'
                else:
                    num = str(temp[0]+1)
                hcd.append(num+suit)
        self.hc = hc
        self.hc_show = hcd

    def get_card(self, idx):
        self.hand[idx] = 1
        self.hand_comm[idx] = 1

class Game:
    def __init__(self, players, deck):
        self.button = 0
        self.round = 'begin'
        self.players = players
        self.num_players = len(self.players)
        self.chips = [0 for _ in range(self.num_players)]
        self.unfold_list = [1 for _ in range(self.num_players)]
        self.community = [0 for _ in range(52)]
        self.comm_show = []
        self.pot = 0
        self.deck = deck
        self.minbet = 2 * MIN_BET
        self.call = 2 * MIN_BET
        self.minraise = 2 * self.minbet
        self.players_pocket = [self.players[i].money for i in range(self.num_players)]
        self.players_origin = [self.players[i].money for i in range(self.num_players)]
        self.max_bet = min(self.players_pocket) - MIN_BET

    def run(self, id, bet):
        # if sum(self.unfold_list) == 1 and self.round != 'pre_flop_bet':
        #     self.round = 'settle'

        if self.round == 'blind':
            # self.button = np.random.randint(low=0, high=self.num_players, size=1)[0]
            self.players[(self.button + 1)%self.num_players].money -= MIN_BET
            self.players[(self.button + 2)%self.num_players].money -= 2 * MIN_BET
            self.chips[(self.button + 1)%self.num_players] = MIN_BET
            self.chips[(self.button + 2)%self.num_players] = 2 * MIN_BET

        elif self.round == 'pre_flop':
            for player in self.players:
                player.get_card(self.deck.dispatch())
                player.get_card(self.deck.dispatch())
            self.max_bet = min(self.players_pocket) - MIN_BET
            return True

        elif self.round == 'pre_flop_bet':
            if not self.unfold_list[id]:
                return True
            else:
                if bet[0] == 'call':
                    difference = self.call - self.chips[id]
                    self.chips[id] = self.call
                    if self.players[id].money < difference:
                        return False
                    else:
                        self.players[id].money -= difference
                        self.players_pocket[id] = self.players[id].money
                        return True
                elif bet[0] == 'raise':
                    if bet[1] - self.call >= self.minraise:
                        difference = bet[1] - self.chips[id]
                        if self.players[id].money < difference:
                            return False
                        elif bet[1] > self.max_bet:
                            return False
                        else:
                            self.chips[id] = bet[1]
                            self.players[id].money -= difference
                            self.players_pocket[id] = self.players[id].money
                            self.call = bet[1]
                            return True
                    else:
                        return False
                elif bet[0] == 'fold':
                    if sum(self.unfold_list) == 1:
                        return False
                    else:
                        self.unfold_list[id] = 0
                        return True
                elif bet[0] == 'check':
                    return False

        elif self.round == 'flop':
            self.pot += sum(self.chips)
            self.chips = [0 for _ in range(self.num_players)]
            self.max_bet = min(self.players_pocket) - MIN_BET
            for _ in range(3):
                num = self.deck.dispatch()
                self.community[num] = 1
                self.comm_show.append(CARD_DICT[num])

            for player in self.players:
                player.renew_handcomm(self.community)

        elif self.round == 'flop_bet':
            if not self.unfold_list[id]:
                return True
            else:
                if bet[0] == 'call':
                    difference = self.call - self.chips[id]
                    if self.players[id].money < difference:
                        return False
                    else:
                        self.chips[id] = self.call
                        self.players[id].money -= difference
                        self.players_pocket[id] = self.players[id].money
                        return True
                elif bet[0] == 'raise':
                    if bet[1] - self.call >= self.minraise:
                        difference = bet[1] - self.chips[id]
                        if self.players[id].money < difference:
                            return False
                        elif bet[1] > self.max_bet:
                            return False
                        else:
                            self.chips[id] = bet[1]
                            self.players[id].money -= difference
                            self.players_pocket[id] = self.players[id].money
                            self.call = bet[1]
                            return True
                    else:
                        return False
                elif bet[0] == 'fold':
                    if sum(self.unfold_list) == 1:
                        return False
                    else:
                        self.unfold_list[id] = 0
                        return True
                elif bet[0] == 'check':
                    return True

        elif self.round == 'turn':
            self.pot += sum(self.chips)
            self.chips = [0 for _ in range(self.num_players)]
            self.max_bet = min(self.players_pocket) - MIN_BET
            num = self.deck.dispatch()
            self.community[num] = 1
            self.comm_show.append(CARD_DICT[num])

            for player in self.players:
                player.renew_handcomm(self.community)

        elif self.round == 'turn_bet':
            if not self.unfold_list[id]:
                return True
            else:
                if bet[0] == 'call':
                    difference = self.call - self.chips[id]
                    if self.players[id].money < difference:
                        return False
                    else:
                        self.chips[id] = self.call
                        self.players[id].money -= difference
                        self.players_pocket[id] = self.players[id].money
                        return True
                elif bet[0] == 'raise':
                    if bet[1] - self.call >= self.minraise:
                        difference = bet[1] - self.chips[id]
                        if self.players[id].money < difference:
                            return False
                        elif bet[1] > self.max_bet:
                            return False
                        else:
                            self.chips[id] = bet[1]
                            self.players[id].money -= difference
                            self.players_pocket[id] = self.players[id].money
                            self.call = bet[1]
                            return True
                    else:
                        return False
                elif bet[0] == 'fold':
                    if sum(self.unfold_list) == 1:
                        return False
                    else:
                        self.unfold_list[id] = 0
                        return True
                elif bet[0] == 'check':
                    return True

        elif self.round == 'river':
            self.pot += sum(self.chips)
            self.chips = [0 for _ in range(self.num_players)]
            self.max_bet = min(self.players_pocket) - MIN_BET
            num = self.deck.dispatch()
            self.community[num] = 1
            self.comm_show.append(CARD_DICT[num])

            for player in self.players:
                player.renew_handcomm(self.community)

        elif self.round == 'river_bet':
            if not self.unfold_list[id]:
                return True
            else:
                if bet[0] == 'call':
                    difference = self.call - self.chips[id]
                    if self.players[id].money < difference:
                        return False
                    else:
                        self.chips[id] = self.call
                        self.players[id].money -= difference
                        self.players_pocket[id] = self.players[id].money
                        return True
                elif bet[0] == 'raise':
                    if bet[1] - self.call >= self.minraise:
                        difference = bet[1] - self.chips[id]
                        if self.players[id].money < difference:
                            return False
                        elif bet[1] > self.max_bet:
                            return False
                        else:
                            self.chips[id] = bet[1]
                            self.players[id].money -= difference
                            self.players_pocket[id] = self.players[id].money
                            self.call = bet[1]
                            return True
                    else:
                        return False
                elif bet[0] == 'fold':
                    if sum(self.unfold_list) == 1:
                        return False
                    else:
                        self.unfold_list[id] = 0
                        return True
                elif bet[0] == 'check':
                    return True

        elif self.round == 'settle':
            if 1 not in self.unfold_list:
                # print('all fold')
                return True
            else:
                maxcard = (0, 'high_card', 6, 4, 3, 2, 1)
                maxid = [-1]
                for id, player in enumerate(self.players):
                    if self.unfold_list[id]:
                        temp = card_checker(player.hc)
                        for x, cont in enumerate(maxcard):
                            if x != 1:
                                if temp[x] > cont:
                                    maxcard = temp
                                    maxid = [id]
                                    break
                                elif temp[x] < cont:
                                    break

                for id in maxid:
                    # print('winner:', id+1)
                    self.players[id].money += self.pot//len(maxid)
                    self.players_pocket[id] = self.players[id].money
            return True


def card_checker(hc):

    suit = dict()
    num = dict()

    # IN SUIT, `0` = HEART, '1' = DIAMOND, '2' = CLUB, '3' = SPADE

    royal_flush = False
    straight_flush = []
    four = []
    full_house = []
    flush = []
    straight = []
    three = []
    pair = []

    for card in hc:
        if num.get(card[0]) is None:
            num[card[0]] = 1
        else:
            num[card[0]] += 1
        if suit.get(card[1]) is None:
            suit[card[1]] = [card[0]]
        else:
            suit[card[1]].append(card[0])

    num_list = list(num.keys())

    # CHECK FLUSH
    for kos in suit:  # kos is kind_of_suit
        if len(suit[kos]) >= 5:
            flush.append(suit[kos])

    # CHECK STRAIGHT
    if len(num_list) >= 5:
        for i in range(len(num_list)-4):
            temp = num_list[i]+1
            arr = [num_list[i]]
            for j in range(4):
                if temp in num_list:
                    arr.append(temp)
                    temp += 1
                else:
                    break
            if len(arr) >= 5:
                straight.append(arr)
        if num_list[0] == 0 and num_list[-1] == 12 and num_list[-2] == 11 and num_list[-3] == 10 and num_list[-4] == 9:
            straight.append([9, 10, 11, 12, 13])

    # CHECK STRAIGHT FLUSH OR ROYAL FLUSH
    if straight and flush:
        for hand in straight:
            if hand in flush:
                straight_flush.append(hand)
        if [9, 10, 11, 12, 13] in straight_flush:
            royal_flush = True

    # CHECK PAIRS
    for hand in num:
        if num[hand] >= 2:
            pair.append(hand)
        if num[hand] >= 3:
            three.append(hand)
        if num[hand] >= 4:
            four.append(hand)

    # CHECK FULL HOUSE
    if three and pair:
        if 0 in three:
            big_three = 13
        else:
            big_three = three[-1]
        pair_copy = pair[:]
        pair_copy.remove(big_three if big_three != 13 else 0)
        if pair_copy:
            if 0 in pair_copy:
                big_pair = 13
            else:
                big_pair = pair_copy[-1]
            full_house.append(big_three)
            full_house.append(big_pair)

    if royal_flush:
        res = (9, 'royal')
        return res
    elif straight_flush:
        res = straight_flush[0]
        if 0 in res:
            high = 13
        else:
            high = res[-1]
        res = (8, 'straight_flush', high)
        return res
    elif four:
        res = (7, 'four', four[0])
        return res
    elif full_house:
        res = (6, 'full_house', full_house[0], full_house[1])
        return res
    elif flush:
        res = flush[0]
        if 0 in res:
            high = 13
        else:
            high = max(res)
        res.remove(high if high != 13 else 0)
        if 0 in res:
            second = 13
        else:
            second = max(res)
        res.remove(second if second != 13 else 0)
        if 0 in res:
            third = 13
        else:
            third = max(res)
        res.remove(third if third != 13 else 0)
        if 0 in res:
            forth = 13
        else:
            forth = max(res)
        res.remove(forth if forth != 13 else 0)
        if 0 in res:
            fifth = 13
        else:
            fifth = max(res)
        res.remove(fifth if fifth != 13 else 0)
        res = (5, 'flush', high, second, third, forth, fifth)
        return res
    elif straight:
        res = straight[-1]
        high = res[-1]
        second = res[-2]
        third = res[-3]
        forth = res[-4]
        fifth = res[-5]
        res = (4, 'straight', high, second, third, forth, fifth)
        return res
    elif three:
        if 0 in three:
            high = 13
        else:
            high = three[-1]
        num_list.remove(high if high != 13 else 0)
        if 0 in num_list:
            second = 13
        else:
            second = num_list[-1]
        num_list.remove(second if second != 13 else 0)
        if 0 in num_list:
            third = 13
        else:
            third = num_list[-1]
        res = (3, 'three', high, second, third)
        return res
    elif pair:
        if 0 in pair:
            high = 13
        else:
            high = pair[-1]
        pair.remove(high if high != 13 else 0)
        num_list.remove(high if high != 13 else 0)
        if pair:
            second = pair[-1]
            num_list.remove(second if second != 13 else 0)
            if 0 in num_list:
                third = 13
            else:
                third = num_list[-1]
            res = (2, 'double_pair', high, second, third)
        else:
            if 0 in num_list:
                second = 13
            else:
                second = num_list[-1]
            num_list.remove(second if second != 13 else 0)
            if 0 in num_list:
                third = 13
            else:
                third = num_list[-1]
            num_list.remove(third if third != 13 else 0)
            if 0 in num_list:
                forth = 13
            else:
                forth = num_list[-1]
            res = (1, 'pair', high, second, third, forth)
        return res
    else:
        if 0 in num_list:
            high = 13
        else:
            high = num_list[-1]
        num_list.remove(high if high != 13 else 0)
        second = num_list[-1]
        num_list.remove(second)
        third = num_list[-1]
        num_list.remove(third)
        forth= num_list[-1]
        num_list.remove(forth)
        fifth = num_list[-1]
        res = (0, 'high_card', high, second, third, forth, fifth)
        return res



if __name__ == '__main__':

    seed = np.random.randint(low=0, high=4096, size=1)[0]
    # seed = 3102
    np.random.seed(seed)
    print('seed:', seed)
    player_num = 6
    money = 1000
    deck = Deck()
    players = ()
    for i in range(player_num):
        player = Player(money)
        players += (player,)

    game = Game(players, deck)
    for stage in PROCESS:
        table(stage, game, 0)

                    # print('stage:', stage, 'id:', id, game.chips, game.players[2].money)

    id = 0
    for player in game.players:
        print('id:', id+1, 'money:', game.players[id].money, game.players[id].hc_show)
        id += 1

        print('best hand:', card_checker(player.hc))

    # seed = np.random.randint(low=0, high=4096, size=1)[0]
    # # seed = 3458
    # print('seed:', seed)
    # np.random.seed(seed)
    # community = [0 for _ in range(52)]
    # dealer = Deck()
    # play1 = Player(1000)
    #
    # for _ in range(5):
    #     community[dealer.dispatch()] = 1
    # for _ in range(2):
    #     play1.get_card(dealer.dispatch())
    #
    # play1.renew_handcomm(community)
    # print('pole combined with community:', play1.hc_show)
    #
    # print('highest ranking combination: ', card_checker(play1.hc))